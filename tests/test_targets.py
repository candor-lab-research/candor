from candor.targets import find_targets, gguf_architecture, gguf_name

HIDDEN = 64


def llama(layers: int = 2, prefix: str = "model.layers") -> dict[str, list[int]]:
    shapes = {}
    for i in range(layers):
        shapes |= {
            f"{prefix}.{i}.self_attn.q_proj.weight": [HIDDEN, HIDDEN],
            f"{prefix}.{i}.self_attn.k_proj.weight": [16, HIDDEN],
            f"{prefix}.{i}.self_attn.o_proj.weight": [HIDDEN, HIDDEN],
            f"{prefix}.{i}.mlp.up_proj.weight": [128, HIDDEN],
            f"{prefix}.{i}.mlp.gate_proj.weight": [128, HIDDEN],
            f"{prefix}.{i}.mlp.down_proj.weight": [HIDDEN, 128],
            f"{prefix}.{i}.input_layernorm.weight": [HIDDEN],
        }
    return shapes


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
    assert found.stack == "model.language_model.layers"
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
    assert found.stack == "model.layers"


def test_fused_experts_without_a_weight_suffix_are_found():
    shapes = llama() | {
        f"model.layers.{i}.feed_forward.experts.down_proj": [8, 128, HIDDEN] for i in range(2)
    }
    found = find_targets(shapes, HIDDEN, 2)
    assert [w.name for w in found.experts] == [
        "model.layers.0.feed_forward.experts.down_proj",
        "model.layers.1.feed_forward.experts.down_proj",
    ]


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
    assert gguf_architecture(exception) != "llama"
