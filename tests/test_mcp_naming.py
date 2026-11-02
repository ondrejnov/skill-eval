from skilleval.mcp.naming import fit_tool_name, match_tool_glob, mcp_tool_name, normalize, server_label


def test_mcp_tool_name_like_claude_code():
    assert mcp_tool_name("webstore-core", "webstore", "orders_get") == "mcp__plugin_webstore-core_webstore__orders_get"


def test_normalize_invalid_chars():
    assert normalize("a.b c:d/é-f_1") == "a_b_c_d__-f_1"
    assert mcp_tool_name("my.plugin", "srv:x", "tool.name") == "mcp__plugin_my_plugin_srv_x__tool_name"
    assert server_label("p", "s") == "plugin_p_s"


def test_fit_tool_name():
    name = "mcp__plugin_" + "x" * 80 + "__tool"
    short = fit_tool_name(name, 64)
    assert len(short) == 64 and short != name
    assert fit_tool_name(name, 64) == short  # deterministické
    assert fit_tool_name("Read", 64) == "Read"


def test_match_tool_glob():
    full = "mcp__plugin_webstore-core_webstore__orders_get"
    assert match_tool_glob("mcp__plugin_webstore-core_webstore__*", full)
    assert match_tool_glob(full, full)
    assert match_tool_glob("mcp__plugin_*__orders_*", full)
    assert not match_tool_glob("mcp__plugin_other_webstore__*", full)
    assert not match_tool_glob("orders_get", full)
    assert match_tool_glob("Bash(npm test *)", "Bash")
    assert not match_tool_glob("read", "Read")
    assert not match_tool_glob("", "Read")
