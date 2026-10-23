import gguf
import pytest

from candor.targets import (
    Writer,
    _converter_table,
    edit_targets,
    find_targets,
    gguf_architecture,
    gguf_name,
    kv_bytes_per_token,
    recurrent_bytes_per_slot,
)

from .helpers import HIDDEN, LLAMA_3B, QWEN35, llama


def test_finds_writers_and_ignores_same_shaped_readers():
    found = find_targets(llama(), HIDDEN, 2)
    assert found.ok
    assert sorted(w.name for w in found.writers) == [
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
        "model.layers.1.self_attn.o_proj.weight",
    ]
    assert found.summary() == {"attn": {"self_attn.o_proj": 2}, "mlp": {"mlp.down_proj": 2}}


def test_hybrid_linear_attention_and_extra_stacks():
    shapes = {}
    for i in range(4):
        prefix = f"model.language_model.layers.{i}"
        if i % 2:
            shapes[f"{prefix}.self_attn.o_proj.weight"] = [HIDDEN, HIDDEN]
        else:
            shapes[f"{prefix}.linear_attn.out_proj.weight"] = [HIDDEN, HIDDEN]
            shapes[f"{prefix}.linear_attn.in_proj_z.weight"] = [HIDDEN, HIDDEN]
        shapes[f"{prefix}.mlp.down_proj.weight"] = [HIDDEN, 128]
    shapes["mtp.layers.0.mlp.down_proj.weight"] = [HIDDEN, 128]
    found = find_targets(shapes, HIDDEN, 4)
    assert found.ok
    assert all(w.name.startswith("model.language_model.layers.") for w in found.writers)
    assert found.summary()["attn"] == {"linear_attn.out_proj": 2, "self_attn.o_proj": 2}
    assert not any(w.name.startswith("mtp.") for w in found.writers)


def test_unknown_writer_is_reported_not_skipped():
    shapes = llama() | {"model.layers.1.mlp.mystery.weight": [HIDDEN, 128]}
    found = find_targets(shapes, HIDDEN, 2)
    assert not found.ok
    assert found.unclassified == ["model.layers.1.mlp.mystery.weight"]


def test_layer_without_an_mlp_writer_is_reported():
    shapes = llama()
    del shapes["model.layers.1.mlp.down_proj.weight"]
    found = find_targets(shapes, HIDDEN, 2)
    assert found.missing == ["layer 1 has no mlp writer"]


def test_too_few_layers_is_reported():
    assert not find_targets(llama(2), HIDDEN, 3).ok


def test_extra_layers_beyond_the_model_are_left_alone():
    found = find_targets(llama(3), HIDDEN, 2)
    assert found.ok
    assert found.extra_layers == [2]
    assert all(w.layer < 2 for w in found.writers)


def test_two_stacks_that_both_fit_are_reported():
    shapes = llama() | llama(prefix="model.other.layers")
    found = find_targets(shapes, HIDDEN, 2)
    assert not found.ok
    assert "several layer stacks" in found.missing[0]


def test_a_vision_stack_of_the_same_depth_is_ignored():
    shapes = llama() | llama(prefix="model.vision_tower.layers")
    found = find_targets(shapes, HIDDEN, 2)
    assert found.ok
    assert all(w.name.startswith("model.layers.") for w in found.writers)


def test_fused_experts_without_a_weight_suffix_are_found():
    shapes = llama() | {
        f"model.layers.{i}.feed_forward.experts.down_proj": [8, 128, HIDDEN] for i in range(2)
    }
    found = find_targets(shapes, HIDDEN, 2)
    assert [w.name for w in found.experts] == [
        "model.layers.0.feed_forward.experts.down_proj",
        "model.layers.1.feed_forward.experts.down_proj",
    ]
    # Stored [experts, in, hidden], as Llama 4's are, which the converter transposes.
    assert found.hidden_last == [w.name for w in found.experts]


def moe(experts: int = 12, layers: int = 2, shared: bool = False) -> dict[str, list[int]]:
    shapes = {}
    for i in range(layers):
        shapes[f"model.layers.{i}.self_attn.o_proj.weight"] = [HIDDEN, HIDDEN]
        shapes[f"model.layers.{i}.mlp.gate.weight"] = [experts, HIDDEN]
        for e in range(experts):
            shapes[f"model.layers.{i}.mlp.experts.{e}.down_proj.weight"] = [HIDDEN, 16]
            shapes[f"model.layers.{i}.mlp.experts.{e}.up_proj.weight"] = [16, HIDDEN]
        if shared:
            shapes[f"model.layers.{i}.mlp.shared_expert.down_proj.weight"] = [HIDDEN, 32]
    return shapes


def test_a_layers_experts_stored_apart_are_one_target_in_expert_order():
    found = find_targets(moe(), HIDDEN, 2)
    assert found.ok and not found.hidden_last
    grouped = edit_targets(found.writers, "qwen3moe", 2)
    experts = grouped["model.layers.0.mlp.experts.down_proj.weight"]
    assert experts.gguf == "blk.0.ffn_down_exps.weight"
    assert experts.names == [f"model.layers.0.mlp.experts.{e}.down_proj.weight" for e in range(12)]
    assert grouped["model.layers.0.self_attn.o_proj.weight"].gguf == "blk.0.attn_output.weight"
    assert len(grouped) == 4


def test_a_shared_expert_is_a_target_of_its_own():
    found = find_targets(moe(experts=4, shared=True), HIDDEN, 2)
    grouped = edit_targets(found.writers, "qwen2moe", 2)
    shared = grouped["model.layers.1.mlp.shared_expert.down_proj.weight"]
    assert shared.gguf == "blk.1.ffn_down_shexp.weight" and len(shared.writers) == 1
    assert grouped["model.layers.1.mlp.experts.down_proj.weight"].gguf == "blk.1.ffn_down_exps.weight"


def test_experts_with_a_number_missing_are_refused():
    shapes = moe(experts=4)
    del shapes["model.layers.1.mlp.experts.2.down_proj.weight"]
    with pytest.raises(ValueError, match="aren't numbered 0 to 2"):
        edit_targets(find_targets(shapes, HIDDEN, 2).writers, "qwen3moe", 2)


def test_experts_held_both_ways_are_refused():
    shapes = moe(experts=1) | {"model.layers.0.mlp.experts.down_proj.weight": [1, HIDDEN, 16]}
    with pytest.raises(ValueError, match="both as one tensor and as experts stored apart"):
        edit_targets(find_targets(shapes, HIDDEN, 2).writers, "qwen3moe", 2)


def test_biases_and_scales_are_not_writers():
    shapes = llama() | {
        "model.layers.0.mlp.down_proj.bias": [HIDDEN, 1],
        "model.layers.0.mlp.down_proj.weight_scale_inv": [HIDDEN, 1],
    }
    assert find_targets(shapes, HIDDEN, 2).ok


def test_norms_on_block_outputs_are_flagged():
    for norm in ("post_feedforward_layernorm", "post_mlp_layernorm"):
        shapes = llama() | {f"model.layers.{i}.{norm}.weight": [HIDDEN] for i in range(2)}
        assert find_targets(shapes, HIDDEN, 2).output_norms
    assert not find_targets(llama(), HIDDEN, 2).output_norms


def test_quantised_writers_are_flagged():
    shapes = llama() | {"model.layers.1.mlp.down_proj.weight_scale": [HIDDEN, 1]}
    found = find_targets(shapes, HIDDEN, 2)
    assert found.quantized == ["model.layers.1.mlp.down_proj.weight"]


def test_gguf_names():
    found = find_targets(llama(), HIDDEN, 2)
    names = {w.name: gguf_name(w, "llama", 2) for w in found.writers}
    assert names["model.layers.1.self_attn.o_proj.weight"] == "blk.1.attn_output.weight"
    assert names["model.layers.0.mlp.down_proj.weight"] == "blk.0.ffn_down.weight"


def test_gguf_names_strip_language_model_like_the_converter():
    found = find_targets(llama(prefix="model.language_model.layers"), HIDDEN, 2)
    assert gguf_name(found.writers[0], "llama", 2) == "blk.0.attn_output.weight"


@pytest.mark.parametrize(
    ("arch", "layers", "name", "expected"),
    [
        ("qwen35", 32, "model.language_model.layers.0.linear_attn.out_proj.weight", "blk.0.ssm_out.weight"),
        ("qwen35", 32, "model.language_model.layers.3.self_attn.o_proj.weight", "blk.3.attn_output.weight"),
        ("qwen35", 32, "model.language_model.layers.31.mlp.down_proj.weight", "blk.31.ffn_down.weight"),
        ("k2-horizon", 36, "model.layers.0.self_attn.o_proj.weight", "blk.0.attn_output.weight"),
        ("k2-horizon", 36, "model.layers.35.mlp.down_proj.weight", "blk.35.ffn_down.weight"),
    ],
)
def test_gguf_names_for_qwen35_and_k2_horizon(arch, layers, name, expected):
    layer = int(name.split(".layers.")[1].split(".")[0])
    assert gguf_name(Writer(name, layer, "attn"), arch, layers) == expected


def test_every_architecture_in_the_converter_table_is_known_to_gguf():
    assert set(_converter_table()["architectures"].values()) <= set(gguf.MODEL_ARCH_NAMES.values())


def test_gguf_architecture_comes_from_the_converter_registry():
    assert gguf_architecture({"architectures": ["LlamaForCausalLM"]}) == "llama"
    assert gguf_architecture({"architectures": ["MistralForCausalLM"]}) == "llama"
    assert gguf_architecture({"architectures": ["Gemma3ForCausalLM"]}) == "gemma3"
    assert gguf_architecture({"architectures": ["Qwen3_5ForConditionalGeneration"]}) == "qwen35"
    assert gguf_architecture({"architectures": ["K2HorizonForCausalLM"]}) == "k2-horizon"
    assert gguf_architecture({"architectures": ["NoSuchModel"]}) is None
    assert gguf_architecture({"model_type": "llama"}) is None


def test_gguf_architecture_prefers_text_config_like_the_converter():
    vision = {"architectures": ["SomeVisionWrapper"], "text_config": {"architectures": ["LlamaForCausalLM"]}}
    assert gguf_architecture(vision) == "llama"
    exception = {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "text_config": {"architectures": ["LlamaForCausalLM"]},
    }
    assert gguf_architecture(exception) == "kimi-k3"


# As a derivative may give it: no layer_types, so llama.cpp goes by full_attention_interval.
QWEN35_BY_INTERVAL = {k: v for k, v in QWEN35.items() if k != "layer_types"}
SMOLLM2 = {"num_hidden_layers": 30, "num_attention_heads": 9, "num_key_value_heads": 3, "hidden_size": 576}


@pytest.mark.parametrize(
    ("text", "arch", "per_token"),
    [
        # As llama.cpp's own log gives them: rude-llama's 112 KiB, Qwen3.5-9B's 32 and K2-Horizon's 144.
        (LLAMA_3B | {"head_dim": 128}, "llama", 112 * 1024),
        (QWEN35, "qwen35", 32 * 1024),
        ({"num_hidden_layers": 36, "num_key_value_heads": 8, "head_dim": 128}, "k2-horizon", 144 * 1024),
        # Without head_dim, as most Llama and Qwen2 configs, it is the hidden size over the heads.
        (LLAMA_3B, "llama", 112 * 1024),
        (SMOLLM2, "llama", 30 * 2 * 3 * 64 * 2),
        # Without num_key_value_heads every head has its own K and V.
        ({"num_hidden_layers": 2, "num_attention_heads": 4, "hidden_size": 64}, "qwen2", 2 * 2 * 4 * 16 * 2),
        (LLAMA_3B | {"head_dim": 128}, "gemma2", None),
        ({"num_hidden_layers": 28, "num_key_value_heads": 8}, "llama", None),
        (QWEN35 | {"layer_types": ["sliding_attention", "full_attention"] * 16}, "qwen35", None),
        # No layer with a KV cache.
        (QWEN35 | {"layer_types": ["linear_attention"] * 32}, "qwen35", None),
        # Heads given per layer: not a layout candor knows, whatever the architecture.
        (LLAMA_3B | {"num_attention_heads": [24] * 28}, "llama", None),
        (LLAMA_3B | {"num_attention_heads": [24] * 28}, "gemma2", None),
        # Without layer_types, llama.cpp makes every fourth Qwen3.5 layer full, or every
        # full_attention_interval-th.
        (QWEN35_BY_INTERVAL, "qwen35", 32 * 1024),
        (
            QWEN35_BY_INTERVAL | {"full_attention_interval": 2},
            "qwen35",
            64 * 1024,
        ),
    ],
)
def test_the_kv_cache_a_token_takes_is_known_only_for_layouts_candor_knows(text, arch, per_token):
    assert kv_bytes_per_token({"text_config": text}, arch, 16) == per_token
    if per_token:
        assert kv_bytes_per_token(text, arch, 8.5) == per_token * 8.5 / 16


def test_a_hybrids_linear_attention_layers_hold_a_state_for_each_slot():
    # 50.25 MiB, as llama.cpp's own log gives Qwen3.5-9B's for one slot.
    assert recurrent_bytes_per_slot({"text_config": QWEN35}, "qwen35") == 24 * (3 * 8192 + 128 * 4096) * 4
    assert recurrent_bytes_per_slot(QWEN35, "qwen35") == 50.25 * 2**20
    assert recurrent_bytes_per_slot(LLAMA_3B, "llama") == 0
    assert recurrent_bytes_per_slot(QWEN35 | {"linear_num_value_heads": None}, "qwen35") is None
    assert recurrent_bytes_per_slot(QWEN35_BY_INTERVAL, "qwen35") == 50.25 * 2**20


def test_kv_bytes_for_moe_architectures():
    model_config = {
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "hidden_size": 256,
    }
    for arch in ("qwen3moe", "olmoe"):
        assert kv_bytes_per_token(model_config | {"head_dim": 64}, arch, 16) == 4 * 2 * 2 * 64 * 2
        assert kv_bytes_per_token(model_config, arch, 16) == 4 * 2 * 2 * 32 * 2
