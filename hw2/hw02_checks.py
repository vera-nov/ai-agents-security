"""Самопроверки ДЗ 2 «Агент с графом знаний».

    from hw02_checks import check
    check("2.1", KnowledgeGraph=KnowledgeGraph)

check за доли секунды прогоняет базовые тесты задания и печатает ✅ или ❌ с причиной. Тесты не
обращаются к настоящей модели: вместо неё ScriptedChatModel — фейковая chat-модель, которая
отвечает заготовленными сообщениями. Её можно использовать и самостоятельно, чтобы отлаживать
код без API.

✅ значит, что прошли базовые тесты. Итоговую оценку ставит проверяющий: он смотрит ещё выводы
ячеек и тексты.

Файл должен лежать рядом с ноутбуком.
"""

from __future__ import annotations

import itertools
import json
import re
import traceback
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

# ---------------------------------------------------------------------------
# Фейковые модели
# ---------------------------------------------------------------------------


class ScriptedChatModel(BaseChatModel):
    """Фейковая chat-модель для тестов без API.

    Отвечает сообщениями из ``responses`` по очереди. Элемент списка — либо
    ``AIMessage``, либо строка (станет ``AIMessage`` с этим текстом), либо функция
    ``(messages) -> AIMessage | str``, если ответ должен зависеть от запроса.
    Когда список закончился, повторяет последний элемент.

    Все полученные запросы сохраняются в ``received``, инструменты из
    ``bind_tools`` — в ``bound_tools``.
    """

    responses: list
    _index: int = PrivateAttr(default=0)
    _received: list = PrivateAttr(default_factory=list)
    _bound_tools: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def received(self) -> list[list[BaseMessage]]:
        return self._received

    @property
    def bound_tools(self) -> list:
        return self._bound_tools

    def bind_tools(self, tools, **kwargs):  # noqa: D401 — сигнатура LangChain
        self._bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._received.append(list(messages))
        item = self.responses[min(self._index, len(self.responses) - 1)]
        self._index += 1
        message = item(messages) if callable(item) else item
        if isinstance(message, str):
            message = AIMessage(content=message)
        message = message.model_copy(deep=True)
        return ChatResult(generations=[ChatGeneration(message=message)])


_call_ids = itertools.count(1)


def tool_call(name: str, **args) -> AIMessage:
    """AIMessage с одним вызовом инструмента ``name(**args)``."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_call_ids)}"}])




# ---------------------------------------------------------------------------
# Инфраструктура
# ---------------------------------------------------------------------------


class CheckFailed(AssertionError):
    pass


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


TOY_SCHEMA = {
    "entity_types": {"Город": "город", "Отель": "отель", "Экскурсия": "экскурсия", "Гид": "гид", "Язык": "язык", "Тур": "тур"},
    "relations": {
        "находится_в": {"source": "Отель|Экскурсия", "target": "Город", "meaning": "где"},
        "работает_в": {"source": "Гид", "target": "Город", "meaning": "где работает"},
        "ведёт": {"source": "Гид", "target": "Экскурсия", "meaning": "проводит"},
        "говорит_на": {"source": "Гид", "target": "Язык", "meaning": "язык"},
        "стартует_от": {"source": "Экскурсия", "target": "Отель", "meaning": "сбор у отеля"},
        "включает": {"source": "Тур", "target": "Экскурсия", "meaning": "в программе"},
        "проживание_в": {"source": "Тур", "target": "Отель", "meaning": "живут"},
    },
}

TOY_DOC = {"doc_id": "e_toy", "title": "Экскурсия «Гастротур»", "text": "«Гастротур» — экскурсия в Казани.\n\nВстречаемся у отеля «Кабан». Ведёт экскурсию Алия."}


def _toy_extractions() -> list[tuple[str, dict]]:
    return [
        ("e_food", {"entities": [{"name": "Гастротур", "type": "Экскурсия"}, {"name": "Казань", "type": "Город"},
                                 {"name": "Алия", "type": "Гид"}, {"name": "Кабан", "type": "Отель"}],
                    "relations": [{"source": "Гастротур", "relation": "находится_в", "target": "Казань", "evidence": "экскурсия в Казани"},
                                  {"source": "Алия", "relation": "ведёт", "target": "Гастротур", "evidence": "Ведёт экскурсию Алия"},
                                  {"source": "Гастротур", "relation": "стартует_от", "target": "Кабан", "evidence": "у отеля «Кабан»"}]}),
        ("g_aliya", {"entities": [{"name": "Алия Гарипова", "type": "Гид"}, {"name": "казань", "type": "Город"},
                                  {"name": "татарский", "type": "Язык"}, {"name": "английский", "type": "Язык"}],
                     "relations": [{"source": "Алия Гарипова", "relation": "работает_в", "target": "казань", "evidence": "в Казани"},
                                   {"source": "Алия Гарипова", "relation": "говорит_на", "target": "татарский", "evidence": "татарский"},
                                   {"source": "Алия Гарипова", "relation": "говорит_на", "target": "английский", "evidence": "английский"}]}),
        ("h_kaban", {"entities": [{"name": "Гостиница «Кабан»", "type": "Отель"}, {"name": "Казань", "type": "Город"}],
                     "relations": [{"source": "Гостиница «Кабан»", "relation": "находится_в", "target": "Казань", "evidence": "в Казани"}]}),
        ("t_weekend", {"entities": [{"name": "Казанские выходные", "type": "Тур"}, {"name": "Гастротур", "type": "Экскурсия"},
                                    {"name": "Петроградка", "type": "Отель"}],
                       "relations": [{"source": "Казанские выходные", "relation": "включает", "target": "Гастротур", "evidence": "Гастротур"},
                                     {"source": "Казанские выходные", "relation": "проживание_в", "target": "Петроградка", "evidence": "Петроградка"}]}),
        ("h_petro", {"entities": [{"name": "Отель «Петроградская сторона»", "type": "Отель"}, {"name": "Санкт-Петербург", "type": "Город"}],
                     "relations": [{"source": "Отель «Петроградская сторона»", "relation": "находится_в", "target": "Санкт-Петербург", "evidence": "в Петербурге"}]}),
    ]


def _toy_graph(KnowledgeGraph):
    kg = KnowledgeGraph()
    for doc_id, extraction in _toy_extractions():
        kg.add_extraction(doc_id, extraction)
    return kg


def _node_by_name(kg, fragment: str, etype: str):
    nodes = [n for n, d in kg.graph.nodes(data=True) if d["type"] == etype and fragment.lower() in d["name"].lower()]
    return nodes


# ---------------------------------------------------------------------------
# Тесты
# ---------------------------------------------------------------------------


def _check_1_1(build_extraction_prompt: Callable, **_):
    messages = build_extraction_prompt(TOY_DOC, TOY_SCHEMA)
    _ensure(isinstance(messages, list) and messages and all(isinstance(m, dict) and {"role", "content"} <= set(m) for m in messages),
            "build_extraction_prompt возвращает список сообщений {'role': ..., 'content': ...}")
    text = "\n".join(m["content"] for m in messages)
    _ensure(all(t in text for t in TOY_SCHEMA["entity_types"]), "В промпте должны быть все типы сущностей схемы")
    _ensure(all(r in text for r in TOY_SCHEMA["relations"]), "В промпте должны быть все типы связей схемы")
    _ensure(TOY_DOC["text"] in text, "В промпте должен быть текст документа целиком")
    _ensure("evidence" in text and "json" in text.lower(), "Промпт просит ответ в JSON и цитату evidence у каждой связи")


def _check_1_2(parse_extraction: Callable, **_):
    good = {"entities": [{"name": "Гастротур", "type": "Экскурсия"}, {"name": "Алия", "type": "Гид"}, {"name": "Казань", "type": "Город"},
                         {"name": "Москва", "type": "Столица"}],
            "relations": [{"source": "Алия", "relation": "ведёт", "target": "Гастротур", "evidence": "Ведёт экскурсию   Алия"},
                          {"source": "Гастротур", "relation": "находится_в", "target": "Казань", "evidence": "экскурсия в Казани"},
                          {"source": "Алия", "relation": "любит", "target": "Гастротур", "evidence": "Ведёт экскурсию Алия"},
                          {"source": "Гастротур", "relation": "ведёт", "target": "Алия", "evidence": "Ведёт экскурсию Алия"},
                          {"source": "Алия", "relation": "работает_в", "target": "Казань", "evidence": "Алия живёт в Казани с детства"},
                          {"source": "Алия", "relation": "говорит_на", "target": "татарский", "evidence": "Алия"}]}
    text = "Вот граф:\n```json\n" + json.dumps(good, ensure_ascii=False) + "\n```"
    result = parse_extraction(text, TOY_DOC["text"], TOY_SCHEMA)
    _ensure(isinstance(result, dict) and isinstance(result.get("entities"), list) and isinstance(result.get("relations"), list),
            "parse_extraction возвращает dict с entities и relations (JSON может быть внутри текста или блока кода)")
    names = {e["name"] for e in result["entities"]}
    _ensure("Москва" not in names and {"Гастротур", "Алия", "Казань"} <= names, "Сущности с типом не из схемы отбрасываются, остальные остаются")
    rels = {(r["source"], r["relation"], r["target"]) for r in result["relations"]}
    _ensure(("Алия", "ведёт", "Гастротур") in rels and ("Гастротур", "находится_в", "Казань") in rels,
            "Корректные связи остаются (цитата сравнивается без учёта лишних пробелов и регистра)")
    _ensure(len(rels) == 2, f"Должны отброситься: связь не из схемы, связь с неподходящими типами концов, цитата, которой нет в документе, "
                            f"конец не среди сущностей. Осталось связей: {len(rels)}")
    for bad in ["", "не json", "{\"entities\": 5}", "{\"entities\": [], \"relations\": [] "]:
        try:
            value = parse_extraction(bad, TOY_DOC["text"], TOY_SCHEMA)
        except Exception as error:  # noqa: BLE001
            raise CheckFailed(f"parse_extraction не должна падать: на {bad!r} упала с {type(error).__name__}") from error
        _ensure(value is None, f"Ответ без корректного JSON → None (вход {bad!r})")


def _check_2_1(KnowledgeGraph: type, **_):
    kg = _toy_graph(KnowledgeGraph)
    guides = _node_by_name(kg, "Алия", "Гид")
    _ensure(len(guides) == 1, f"«Алия» и «Алия Гарипова» — одна вершина-гид, а получилось {len(guides)}")
    _ensure(kg.graph.nodes[guides[0]]["name"] == "Алия Гарипова", "Имя вершины — самое полное из встреченных («Алия Гарипова»)")
    _ensure(len([n for n, d in kg.graph.nodes(data=True) if d["type"] == "Город" and d["name"].lower() == "казань"]) == 1,
            "«Казань» и «казань» — одна вершина-город")
    _ensure(len(_node_by_name(kg, "Кабан", "Отель")) == 1, "«Кабан» и «Гостиница «Кабан»» — одна вершина-отель")
    petro = [n for n, d in kg.graph.nodes(data=True) if d["type"] == "Отель" and "петроград" in d["name"].lower()]
    _ensure(len(petro) == 1, "«Петроградка» и «Отель «Петроградская сторона»» — одна вершина-отель")
    _ensure(kg.resolve("Москва", "Город") is None, "resolve возвращает None для сущности, которой в графе нет")
    _ensure(kg.resolve("Алия", "Экскурсия") is None, "resolve учитывает тип: гид «Алия» не совпадает с экскурсией")
    kg2 = KnowledgeGraph()
    for doc_id, extraction in reversed(_toy_extractions()):
        kg2.add_extraction(doc_id, extraction)
    _ensure(len(_node_by_name(kg2, "Алия", "Гид")) == 1, "Слияние не должно зависеть от порядка документов")


def _check_2_2(KnowledgeGraph: type, **_):
    kg = _toy_graph(KnowledgeGraph)
    guide = _node_by_name(kg, "Алия", "Гид")[0]
    edges = [d for _, _, d in kg.graph.out_edges(guide, data=True)]
    _ensure({"ведёт", "говорит_на", "работает_в"} <= {d["relation"] for d in edges}, "Связи из разных документов сходятся на одной вершине")
    _ensure(all({"relation", "doc_id", "evidence"} <= set(d) for d in edges), "У каждого ребра есть relation, doc_id и evidence")
    leads = [d for d in edges if d["relation"] == "ведёт"]
    _ensure(len(leads) == 1 and leads[0]["doc_id"] == "e_food", "doc_id ребра — документ, из которого связь извлечена")
    _ensure(set(kg.graph.nodes[guide]["doc_ids"]) == {"e_food", "g_aliya"}, "У вершины хранятся doc_ids всех документов, где она встретилась")
    before = kg.graph.number_of_edges()
    kg.add_extraction(*_toy_extractions()[0])
    _ensure(kg.graph.number_of_edges() == before, "Повторное добавление того же документа не создаёт дублей рёбер")


def _check_2_3(KnowledgeGraph: type, **_):
    kg = _toy_graph(KnowledgeGraph)
    found = kg.find("гастротур")
    _ensure(found and kg.graph.nodes[found[0]]["type"] == "Экскурсия", "find('гастротур') находит экскурсию «Гастротур»")
    found = kg.find("На каких языках говорит Алия Гарипова?")
    _ensure(found and kg.graph.nodes[found[0]]["type"] == "Гид", "find по вопросу целиком находит упомянутую в нём сущность")
    _ensure(kg.find("Казань", etype="Отель") == [], "Фильтр etype оставляет только вершины этого типа")
    _ensure(kg.find("сноуборд") == [], "Если ничего не похоже — пустой список")


def _check_2_4(KnowledgeGraph: type, **_):
    kg = _toy_graph(KnowledgeGraph)
    tour = _node_by_name(kg, "выходные", "Тур")[0]
    one = kg.expand([tour], depth=1)
    _ensure(isinstance(one, list) and all({"source", "relation", "target", "doc_id", "evidence"} <= set(e) for e in one),
            "expand возвращает список рёбер-словарей source, relation, target, doc_id, evidence")
    _ensure({e["relation"] for e in one} == {"включает", "проживание_в"}, "depth=1 — только рёбра, касающиеся стартовых вершин")
    two = kg.expand([tour], depth=2)
    _ensure("ведёт" in {e["relation"] for e in two}, "depth=2 доходит до гида экскурсии (рёбра обходятся в обе стороны)")
    three = kg.expand([tour], depth=3)
    _ensure("говорит_на" in {e["relation"] for e in three}, "depth=3 доходит до языков гида")
    _ensure(len(three) == len({(e["source"], e["relation"], e["target"], e["doc_id"]) for e in three}), "Рёбра в результате не повторяются")
    only = kg.expand([tour], depth=3, relations=["включает", "ведёт"])
    _ensure({e["relation"] for e in only} <= {"включает", "ведёт"} and only, "relations ограничивает, по каким связям идти")
    _ensure(len(kg.expand([tour], depth=3, max_edges=2)) == 2, "max_edges ограничивает размер подграфа")


def _check_2_5(KnowledgeGraph: type, **_):
    kg = _toy_graph(KnowledgeGraph)
    guide = _node_by_name(kg, "Алия", "Гид")[0]
    text = kg.evidence(kg.expand([guide], depth=1))
    _ensure(isinstance(text, str) and "Алия Гарипова" in text and "татарский" in text, "evidence — текст с именами вершин и связями")
    _ensure("[g_aliya]" in text and "[e_food]" in text, "У каждой связи указан [doc_id] источника")
    _ensure("Ведёт экскурсию Алия" in text, "У связи есть цитата-обоснование из документа")
    empty = kg.evidence([])
    _ensure(isinstance(empty, str) and empty.strip(), "Для пустого подграфа — понятный текст, а не пустая строка")


def _check_3_1(make_graph_tool: Callable, KnowledgeGraph: type, **_):
    kg = _toy_graph(KnowledgeGraph)
    tool = make_graph_tool(kg)
    _ensure(isinstance(tool, BaseTool) and tool.name == "graph_search", "make_graph_tool возвращает инструмент graph_search")
    schema = convert_to_openai_tool(tool)["function"]
    props = schema["parameters"]["properties"]
    _ensure("entities" in props and props["entities"].get("type") == "array", "Аргумент entities — список названий сущностей")
    _ensure("depth" in props, "Есть аргумент depth — глубина поиска")
    _ensure(len(schema.get("description", "")) >= 80, "Описание инструмента объясняет модели, когда искать по графу, а когда нет")
    text = tool.invoke({"entities": ["Казанские выходные"], "depth": 3})
    _ensure("Алия Гарипова" in text and "[e_food]" in text, "Результат — связи подграфа с [doc_id]")
    text = tool.invoke({"entities": ["Ночной Стамбул"], "depth": 1})
    _ensure(isinstance(text, str) and re.search(r"не найден|нет в графе|ничего", text, re.IGNORECASE), "Если сущности нет в графе — прямо сказать модели")
    try:
        text = tool.invoke({"entities": ["Гастротур"], "depth": 7})
    except Exception as error:  # noqa: BLE001
        raise CheckFailed(f"Неверная глубина не должна ронять инструмент: {type(error).__name__}") from error


_ids = itertools.count(1)


def _call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}])


def _check_3_2(build_agent: Callable, ask: Callable, AGENT_PROMPT: str, **_):
    from langchain_core.tools import StructuredTool

    _ensure(isinstance(AGENT_PROMPT, str) and len(AGENT_PROMPT) >= 150, "AGENT_PROMPT слишком короткий: когда какой поиск, цитирование, отказ")
    _ensure("graph_search" in AGENT_PROMPT and "text_search" in AGENT_PROMPT, "Промпт объясняет, когда использовать text_search, а когда graph_search")
    text_tool = StructuredTool.from_function(lambda query: "[h_kaban] Гостиница «Кабан» … 5100 рублей", name="text_search", description="поиск по документам")
    graph_tool = StructuredTool.from_function(lambda entities, depth=2: "Алия Гарипова —говорит_на→ татарский [g_aliya]",
                                              name="graph_search", description="поиск по графу")
    llm = ScriptedChatModel(responses=[_call("graph_search", entities=["Гастротур"], depth=2),
                                       _call("text_search", query="Кабан цена"),
                                       AIMessage(content="На татарском [g_aliya]; ночь 5100 рублей [h_kaban].")])
    agent = build_agent(llm, [text_tool, graph_tool])
    result = ask(agent, "вопрос")
    _ensure(isinstance(result, dict) and {"answer", "messages", "tool_calls", "cited_doc_ids", "llm_calls"} <= set(result),
            "ask возвращает dict с answer, messages, tool_calls, cited_doc_ids, llm_calls")
    _ensure([name for name, _ in result["tool_calls"]] == ["graph_search", "text_search"], "tool_calls — (имя, аргументы) вызовов по порядку")
    _ensure(result["cited_doc_ids"] == ["g_aliya", "h_kaban"], "cited_doc_ids — doc_id из ответа (extract_citations)")
    _ensure(result["llm_calls"] == 3, f"llm_calls — число ответов модели, ожидалось 3, получено {result['llm_calls']}")
    system = llm.received[0][0]
    _ensure(system.type == "system" and AGENT_PROMPT in system.content, "build_agent передаёт AGENT_PROMPT в create_agent")
    llm = ScriptedChatModel(responses=[lambda messages: _call("text_search", query="снова")])
    try:
        result = ask(build_agent(llm, [text_tool, graph_tool]), "зациклись", max_steps=3)
    except Exception as error:  # noqa: BLE001
        raise CheckFailed(f"ask не должен падать при превышении лимита шагов: {type(error).__name__}") from error
    _ensure(len(llm.received) == 3 and result["answer"], "max_steps=3 — не больше трёх обращений к модели и понятный ответ")


_CHECKS = {"1.1": _check_1_1, "1.2": _check_1_2, "2.1": _check_2_1, "2.2": _check_2_2, "2.3": _check_2_3, "2.4": _check_2_4,
           "2.5": _check_2_5, "3.1": _check_3_1, "3.2": _check_3_2}


def check(task: str, **objects) -> None:
    """Запускает тесты задания ``task``. Печатает ✅ или ❌ с причиной."""
    if task not in _CHECKS:
        raise KeyError(f"Нет такого задания: {task}. Есть: {', '.join(_CHECKS)}")
    try:
        _CHECKS[task](**objects)
    except CheckFailed as error:
        print(f"❌ Задание {task}: {error}")
        return
    except Exception as error:  # noqa: BLE001
        print(f"❌ Задание {task}: код упал с {type(error).__name__}: {error}")
        traceback.print_exc(limit=3)
        return
    print(f"✅ Задание {task}: базовые тесты пройдены")
