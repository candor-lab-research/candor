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


def test_wrong_layer_count_is_reported():
    assert not find_targets(llama(3), HIDDEN, 2).ok


def test_gguf_names():
    found = find_targets(llama(), HIDDEN, 2)
    names = {w.name: gguf_name(w, "llama") for w in found.writers}
    assert names["model.layers.1.self_attn.o_proj.weight"] == "blk.1.attn_output.weight"
    assert names["model.layers.0.mlp.down_proj.weight"] == "blk.0.ffn_down.weight"


def test_gguf_names_strip_language_model_like_the_converter():
    found = find_targets(llama(prefix="model.language_model.layers"), HIDDEN, 2)
    assert gguf_name(found.writers[0], "llama") is not None


def test_gguf_architecture():
    assert gguf_architecture("llama") == "llama"
    assert gguf_architecture("qwen3_5") == "qwen35"
    assert gguf_architecture("k2_horizon") == "k2-horizon"
    assert gguf_architecture("no_such_model") is None
