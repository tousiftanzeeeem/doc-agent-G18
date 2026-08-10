from doc_agent.agent import tools

def test_registry_is_tool_subclasses():
    for t in tools.REGISTRY:
        assert issubclass(t, tools.Tool) and isinstance(t.name, str)
