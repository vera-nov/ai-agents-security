"""Автопроверки ДЗ 1 «Мой первый агент».

Как пользоваться в ноутбуке:

    from hw01_checks import check
    check("2.2", search_routes=search_routes)

``check`` за доли секунды прогоняет тесты задания и печатает ✅ или ❌ с причиной.
Тесты не обращаются к настоящей LLM: где нужна модель, используется
``ScriptedChatModel`` — фейковая модель, которая отвечает заранее заданными
сообщениями. Её можно использовать и самостоятельно, чтобы отлаживать свой код
без трат на API.

✅ означает, что прошли базовые тесты. Итоговую оценку ставит проверяющий
по рубрике: он смотрит ещё качество описаний, трейсы и выводы.

Файл должен лежать рядом с ноутбуком (в Colab его нужно загрузить в /content).
"""

from __future__ import annotations

import itertools
import json
import re
import traceback
from pathlib import Path
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

DATA_PATH = Path(__file__).with_name("hw01_travel_data.json")


# ---------------------------------------------------------------------------
# Фейковая модель
# ---------------------------------------------------------------------------


class ScriptedChatModel(BaseChatModel):
    """Фейковая chat-модель для тестов без API.

    Отвечает сообщениями из ``responses`` по очереди. Элемент списка — либо
    ``AIMessage``, либо функция ``(messages) -> AIMessage``, если ответ должен
    зависеть от истории. Когда список закончился, повторяет последний элемент.

    Все полученные запросы сохраняются в ``received`` (список списков сообщений),
    а инструменты из ``bind_tools`` — в ``bound_tools``.

    Пример::

        llm = ScriptedChatModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "get_city_info",
                                               "args": {"city": "Казань"},
                                               "id": "call_1"}]),
            AIMessage(content="Казань находится в России."),
        ])
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
        # Копия, чтобы повторные ответы не делили один объект
        message = message.model_copy(deep=True)
        return ChatResult(generations=[ChatGeneration(message=message)])


_call_ids = itertools.count(1)


def tool_call(name: str, **args) -> AIMessage:
    """AIMessage с одним вызовом инструмента ``name(**args)``."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_call_ids)}"}])


def endless_tool_calls(name: str, **args) -> Callable[[list], AIMessage]:
    """Ответ-функция: модель бесконечно зовёт один и тот же инструмент."""
    return lambda messages: tool_call(name, **args)


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------


class CheckFailed(AssertionError):
    pass


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def _load_data() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def _schema(tool_like: Any) -> dict:
    """OpenAI-схема инструмента: parameters, name, description."""
    if isinstance(tool_like, dict):
        return tool_like["function"]
    return convert_to_openai_tool(tool_like)["function"]


def _prop(schema: dict, name: str) -> dict:
    props = schema.get("parameters", {}).get("properties", {})
    _ensure(name in props, f"В схеме нет аргумента `{name}`. Есть: {list(props)}")
    prop = dict(props[name])
    # Optional[...] в pydantic превращается в anyOf: [..., {"type": "null"}]
    for variant in prop.get("anyOf", []):
        if variant.get("type") != "null":
            prop = {**variant, **{k: v for k, v in prop.items() if k != "anyOf"}}
    return prop


def _numbers(text: str) -> list[float]:
    return [float(x.replace(",", ".")) for x in re.findall(r"-?\d+(?:[.,]\d+)?", text.replace(" ", ""))]


def _invoke_text(tool: BaseTool, args: dict) -> str:
    result = tool.invoke(args)
    return result if isinstance(result, str) else str(result)


def _raises_or_reports_error(tool: BaseTool, args: dict) -> bool:
    """True, если инструмент сообщил об ошибке: ToolException или текст ошибки."""
    try:
        text = _invoke_text(tool, args)
    except ToolException:
        return True
    except Exception:  # noqa: BLE001 — ValueError и прочее считаем «не так»
        return False
    return bool(re.search(r"ошибк|неизвестн|не найден|нет так|error|unknown", text, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Тесты по заданиям
# ---------------------------------------------------------------------------


def _check_1_1(schema: dict, **_):
    _ensure(isinstance(schema, dict), "Схема должна быть словарём (dict)")
    _ensure(schema.get("type") == "function", 'Верхний уровень: {"type": "function", "function": {...}}')
    fn = schema.get("function", {})
    _ensure(fn.get("name") == "convert_currency", "Имя функции должно быть `convert_currency`")
    _ensure(len(fn.get("description", "")) >= 20, "Добавь содержательное описание (description) инструмента")
    params = fn.get("parameters", {})
    _ensure(params.get("type") == "object", 'parameters должен иметь "type": "object"')
    required = set(params.get("required", []))
    _ensure(required == {"amount", "from_currency", "to_currency"}, f"required должен содержать все три аргумента, сейчас: {sorted(required)}")
    _ensure(_prop(fn, "amount").get("type") == "number", "amount должен иметь тип number")
    for name in ("from_currency", "to_currency"):
        prop = _prop(fn, name)
        _ensure(prop.get("type") == "string", f"{name} должен иметь тип string")
        _ensure(prop.get("description"), f"У аргумента {name} нет description — модель не узнает формат (ISO-код)")


def _check_1_2(run_manual_agent: Callable, schema: dict, registry: dict, **_):
    # 1) Обычный сценарий: один вызов инструмента, потом текст
    llm = ScriptedChatModel(responses=[
        tool_call("convert_currency", amount=100, from_currency="EUR", to_currency="RUB"),
        AIMessage(content="100 евро — это 10000 рублей."),
    ])
    result = run_manual_agent(llm, "Сколько рублей в 100 евро?", [schema], registry, max_steps=5)
    _ensure(isinstance(result, dict), "run_manual_agent должен возвращать dict")
    _ensure({"answer", "messages", "steps"} <= set(result), "В результате нужны ключи answer, messages, steps")
    _ensure(result["answer"] == "100 евро — это 10000 рублей.", "answer должен быть текстом последнего ответа модели")
    _ensure(llm.bound_tools, "Модель должна получить инструменты через llm.bind_tools(...)")
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    _ensure(len(tool_messages) == 1, "После вызова инструмента в истории должен появиться ровно один ToolMessage")
    call_id = llm.received[1][-2].tool_calls[0]["id"] if len(llm.received) > 1 else None
    _ensure(tool_messages[0].tool_call_id == call_id, "ToolMessage.tool_call_id должен совпадать с id вызова из AIMessage")
    _ensure("10000" in tool_messages[0].content.replace(" ", ""), "В ToolMessage должен быть результат функции convert_currency")
    _ensure(isinstance(llm.received[1][-1], ToolMessage), "Второй запрос к модели должен заканчиваться ToolMessage с результатом")
    _ensure(result["steps"] == 2, f"steps — число обращений к модели; ожидалось 2, получено {result['steps']}")

    # 2) Ошибка, потом успешный вызов: модель исправилась, счётчик обнулился, цикл продолжается
    llm = ScriptedChatModel(responses=[
        tool_call("convert_currency", amount=5, from_currency="XXX", to_currency="RUB"),
        tool_call("convert_currency", amount=5, from_currency="EUR", to_currency="RUB"),
        tool_call("book_hotel", city="Ереван"),
        AIMessage(content="Готово."),
    ])
    try:
        result = run_manual_agent(llm, "q", [schema], registry, max_steps=6, max_tool_errors=2)
    except TypeError as error:
        raise CheckFailed(f"У run_manual_agent должен быть параметр max_tool_errors: {error}") from error
    except Exception as error:  # noqa: BLE001
        raise CheckFailed(f"Ошибка инструмента уронила цикл: {type(error).__name__}: {error}. Её нужно вернуть модели в ToolMessage") from error
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    _ensure(len(tool_messages) == 3, "На каждый вызов (даже ошибочный) нужен ToolMessage")
    _ensure("XXX" in tool_messages[0].content, "Текст ошибки инструмента должен попасть в ToolMessage, чтобы модель могла исправиться")
    _ensure("book_hotel" in tool_messages[2].content, "Для неизвестного инструмента верни модели сообщение с его именем")
    _ensure(result["answer"] == "Готово.", "Ошибки, разделённые успешным вызовом, — не «подряд»: успешный вызов должен обнулять счётчик, и цикл должен дойти до ответа модели")

    # 3) Две ошибки подряд (в том числе несуществующий инструмент) → остановка
    llm = ScriptedChatModel(responses=[
        tool_call("convert_currency", amount=5, from_currency="XXX", to_currency="RUB"),
        tool_call("book_hotel", city="Ереван"),
        AIMessage(content="Не должно сюда дойти."),
    ])
    result = run_manual_agent(llm, "q", [schema], registry, max_steps=6, max_tool_errors=2)
    _ensure(len(llm.received) == 2, f"После max_tool_errors=2 ошибок подряд цикл должен остановиться: модель вызвана {len(llm.received)} раз(а) вместо 2. Несуществующий инструмент тоже считается ошибкой")
    _ensure(isinstance(result["answer"], str) and result["answer"] and result["answer"] != "Не должно сюда дойти.", "При остановке из-за ошибок верни в answer понятное сообщение")
    _ensure("book_hotel" in result["answer"], "В answer при остановке должен быть текст последней ошибки")
    _ensure(result["steps"] == 2, f"steps — число обращений к модели; ожидалось 2, получено {result['steps']}")

    llm = ScriptedChatModel(responses=[
        tool_call("convert_currency", amount=5, from_currency="XXX", to_currency="RUB"),
        tool_call("convert_currency", amount=5, from_currency="YYY", to_currency="RUB"),
        AIMessage(content="Не нашёл такую валюту."),
    ])
    result = run_manual_agent(llm, "q", [schema], registry, max_steps=6, max_tool_errors=3)
    _ensure(result["answer"] == "Не нашёл такую валюту.", "Лимит ошибок должен браться из параметра max_tool_errors (при 3 две ошибки подряд ещё не повод останавливаться)")

    # 4) Лимит шагов
    llm = ScriptedChatModel(responses=[endless_tool_calls("convert_currency", amount=1, from_currency="EUR", to_currency="USD")])
    result = run_manual_agent(llm, "q", [schema], registry, max_steps=3)
    _ensure(len(llm.received) == 3, f"При max_steps=3 модель должна быть вызвана 3 раза, вызвана {len(llm.received)}")
    _ensure(isinstance(result["answer"], str) and result["answer"], "При превышении лимита верни понятный текст в answer, а не None")


def _check_2_1(get_city_info: BaseTool, **_):
    _ensure(isinstance(get_city_info, BaseTool), "get_city_info должен быть инструментом LangChain (@tool)")
    schema = _schema(get_city_info)
    _ensure(schema["name"] == "get_city_info", "Имя инструмента: get_city_info")
    _ensure(len(schema.get("description", "")) >= 20, "Нужен содержательный docstring")
    _ensure("Args:" not in schema.get("description", ""), "Описания аргументов попали в description инструмента. Используй @tool(parse_docstring=True)")
    _ensure(_prop(schema, "city").get("description"), "У аргумента city нет описания. Используй @tool(parse_docstring=True) и раздел Args в docstring")
    text = _invoke_text(get_city_info, {"city": "стамбул"})
    for part in ("Турция", "TRY"):
        _ensure(part in text, f"Ответ для «стамбул» должен содержать «{part}» (регистр названия не должен мешать)")
    _ensure("Россия" in _invoke_text(get_city_info, {"city": "Питер"}), "Должны работать алиасы (Питер → Санкт-Петербург): используй normalize_city")
    _ensure(_raises_or_reports_error(get_city_info, {"city": "Атлантида"}), "Для неизвестного города брось ToolException")


def _check_2_2(search_routes: BaseTool, RouteQuery: type, **_):
    _ensure(isinstance(search_routes, BaseTool), "search_routes должен быть инструментом LangChain")
    _ensure(search_routes.args_schema is RouteQuery, "Передай RouteQuery через args_schema=")
    schema = _schema(search_routes)
    _ensure(schema["name"] == "search_routes", "Имя инструмента: search_routes")
    required = set(schema["parameters"].get("required", []))
    _ensure(required == {"origin", "destination"}, f"Обязательными должны быть только origin и destination, сейчас: {sorted(required)}")
    transport = _prop(schema, "transport")
    _ensure(set(transport.get("enum", [])) == {"plane", "train", "bus", "any"}, "transport: Literal['plane', 'train', 'bus', 'any']")
    _ensure(transport.get("default") == "any", "У transport должно быть значение по умолчанию 'any'")
    price = _prop(schema, "max_price_eur")
    _ensure(price.get("minimum") == 0 or price.get("exclusiveMinimum") == 0, "max_price_eur должен быть ≥ 0 (Field(ge=0))")
    for name in ("origin", "destination", "max_price_eur"):
        _ensure(_prop(schema, name).get("description"), f"У аргумента {name} нет description")

    text = _invoke_text(search_routes, {"origin": "Санкт-Петербург", "destination": "Москва"})
    _ensure(all(p in text for p in ("40", "60", "20")), "Маршруты двусторонние: СПб → Москва должен найти поезд, самолёт и автобус")
    _ensure(text.find("20") < text.find("40") < text.find("60"), "Маршруты должны быть отсортированы по цене (дешёвые первыми)")
    text = _invoke_text(search_routes, {"origin": "Москва", "destination": "Санкт-Петербург", "transport": "train"})
    _ensure("40" in text and "60" not in text, "Фильтр transport не работает")
    text = _invoke_text(search_routes, {"origin": "Москва", "destination": "Санкт-Петербург", "max_price_eur": 30})
    _ensure("20" in text and "40" not in text, "Фильтр max_price_eur не работает")
    text = _invoke_text(search_routes, {"origin": "Москва", "destination": "Белград"})
    _ensure(re.search(r"не найден|нет", text, re.IGNORECASE), "Когда прямых маршрутов нет, верни понятный текст (а не пустую строку)")
    _ensure(_raises_or_reports_error(search_routes, {"origin": "Москва", "destination": "Атлантида"}), "Для неизвестного города брось ToolException")


def _check_2_3(convert_currency_tool: BaseTool, convert_currency: Callable, **_):
    _ensure(isinstance(convert_currency_tool, StructuredTool), "Используй StructuredTool.from_function(...)")
    _ensure(convert_currency_tool.func is convert_currency, "Оберни существующую функцию convert_currency, не переписывая её")
    schema = _schema(convert_currency_tool)
    _ensure(schema["name"] == "convert_currency", "Имя инструмента: convert_currency")
    _ensure(len(schema.get("description", "")) >= 20, "Передай содержательное description")
    _ensure("10000" in _invoke_text(convert_currency_tool, {"amount": 100, "from_currency": "EUR", "to_currency": "RUB"}).replace(" ", ""), "Инструмент должен вызывать convert_currency")


def _check_2_4(ItineraryTool: type, **_):
    data = _load_data()
    _ensure(isinstance(ItineraryTool, type) and issubclass(ItineraryTool, BaseTool), "ItineraryTool должен наследоваться от BaseTool")
    tool = ItineraryTool(data=data)
    schema = _schema(tool)
    _ensure(schema["name"] == "manage_itinerary", "Имя инструмента: manage_itinerary")
    _ensure(set(_prop(schema, "action").get("enum", [])) == {"add", "show", "clear"}, "action: Literal['add', 'show', 'clear']")
    nights = _prop(schema, "nights")
    _ensure(nights.get("minimum") == 1 and nights.get("maximum") == 30, "nights: от 1 до 30 (Field(ge=1, le=30))")

    tool.invoke({"action": "add", "city": "Тбилиси", "nights": 3})
    tool.invoke({"action": "add", "city": "ереван", "nights": 2})
    text = _invoke_text(tool, {"action": "show"})
    # Тбилиси: 3 * 150 GEL / 3 = 150 EUR; Ереван: 2 * 25000 AMD / 420 ≈ 119.05 EUR
    _ensure("Итого ночей: 5" in text, f"show должен содержать строку «Итого ночей: 5». Получено:\n{text}")
    match = re.search(r"Стоимость отелей:\s*≈\s*([\d\s.,]+?)\s*EUR", text)
    _ensure(match is not None, "show должен содержать стоимость отелей в формате «Стоимость отелей: ≈ 269.05 EUR»")
    _ensure(abs(_numbers(match.group(1))[0] - 269.05) < 0.02, f"Стоимость отелей посчитана неверно: ожидалось ≈ 269.05 EUR, получено {match.group(0)}")
    _ensure(text.find("Тбилиси") < text.find("Ереван"), "Города в show — в порядке добавления")

    other = ItineraryTool(data=data)
    _ensure("Итого ночей: 0" in _invoke_text(other, {"action": "show"}), "Состояние должно быть своим у каждого экземпляра (PrivateAttr), а не общим на класс")
    tool.invoke({"action": "clear"})
    _ensure("Итого ночей: 0" in _invoke_text(tool, {"action": "show"}), "clear должен очищать маршрут")
    _ensure(_raises_or_reports_error(tool, {"action": "add", "city": "Атлантида", "nights": 2}), "add с неизвестным городом → ToolException")
    _ensure(_raises_or_reports_error(tool, {"action": "add", "city": "Казань"}), "add без nights → ToolException")


def _check_2_5(get_climate: BaseTool, **_):
    _ensure(isinstance(get_climate, BaseTool), "get_climate должен быть инструментом LangChain")
    schema = _schema(get_climate)
    _ensure(schema["name"] == "get_climate", "Имя инструмента: get_climate")
    _ensure(len(schema.get("description", "")) >= 20, "Нужен содержательный docstring")
    city, month = _prop(schema, "city"), _prop(schema, "month")
    _ensure(city.get("type") == "string" and month.get("type") == "integer", "Аннотируй типы: city: str, month: int")
    _ensure(city.get("description") and month.get("description"), "У обоих аргументов должны быть описания")
    _ensure(month.get("minimum") == 1 and month.get("maximum") == 12, "month ограничь диапазоном 1..12")
    text = _invoke_text(get_climate, {"city": "Дубай", "month": 7})
    _ensure("35" in text, "Для Дубая в июле ожидается 35 °C")
    _ensure(_raises_or_reports_error(get_climate, {"city": "Атлантида", "month": 1}), "Для неизвестного города брось ToolException")


def _check_3(build_agent: Callable, chat: Callable, tools: list, SYSTEM_PROMPT: str, **_):
    _ensure(isinstance(SYSTEM_PROMPT, str) and len(SYSTEM_PROMPT) >= 200, "SYSTEM_PROMPT слишком короткий: опиши роль, правила работы с инструментами и формат ответа")

    llm = ScriptedChatModel(responses=[
        tool_call("get_city_info", city="Казань"),
        AIMessage(content="Казань — столица Татарстана."),
        AIMessage(content="Вы спрашивали про Казань."),
        AIMessage(content="Я пока ничего о вас не знаю."),
    ])
    agent = build_agent(llm, tools)
    result = chat(agent, [], "Расскажи про Казань")
    _ensure(isinstance(result, tuple) and len(result) == 2, "chat должен возвращать пару (answer, history)")
    answer, history = result
    _ensure(answer == "Казань — столица Татарстана.", "answer — текст последнего ответа агента")
    _ensure(isinstance(history, list) and all(isinstance(m, BaseMessage) for m in history), "history — список сообщений LangChain (result['messages'])")
    _ensure(any(isinstance(m, ToolMessage) for m in history), "В истории должны остаться вызов инструмента и его результат")
    _ensure(any(getattr(t, "name", None) == "get_city_info" for t in llm.bound_tools), "build_agent должен передать инструменты агенту")
    system = llm.received[0][0]
    _ensure(system.type == "system" and SYSTEM_PROMPT in system.content, "build_agent должен передать SYSTEM_PROMPT в create_agent(system_prompt=...)")

    before = list(history)
    _, history2 = chat(agent, history, "О каком городе я спрашивал?")
    _ensure(history == before, "chat не должен изменять переданный список history")
    _ensure(any(isinstance(m, HumanMessage) and "Казань" in m.content for m in llm.received[2]), "Агент должен видеть прошлые реплики: передайте history вместе с новым сообщением")
    _ensure(len(history2) > len(history), "Обновлённая история должна содержать и старые, и новые сообщения")

    chat(agent, [], "Что ты обо мне знаешь?")
    _ensure(not any("Казань" in str(m.content) for m in llm.received[3]), "С пустой историей агент не должен видеть прошлый диалог")


def _check_4_1(tools: list, build_agent: Callable, **_):
    for tool in tools:
        _ensure(getattr(tool, "handle_tool_error", False), f"У инструмента {tool.name} не задан handle_tool_error: ToolException уронит агента")
    llm = ScriptedChatModel(responses=[tool_call("get_city_info", city="Атлантида"), AIMessage(content="Такого города нет в базе.")])
    agent = build_agent(llm, tools)
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": "Расскажи про Атлантиду"}]})
    except Exception as error:  # noqa: BLE001
        raise CheckFailed(f"Агент упал на ошибке инструмента: {type(error).__name__}: {error}") from error
    errors = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    _ensure(errors and "Атлантида" in errors[0].content, "Текст ошибки должен объяснять модели, что не так (упомяни город)")


def _check_4_2(safe_invoke: Callable, build_agent: Callable, tools: list, **_):
    llm = ScriptedChatModel(responses=[endless_tool_calls("get_city_info", city="Казань")])
    agent = build_agent(llm, tools)
    old_history = [HumanMessage(content="Привет"), AIMessage(content="Здравствуйте!")]
    try:
        result = safe_invoke(agent, old_history, "Зациклись", max_steps=4)
    except Exception as error:  # noqa: BLE001
        raise CheckFailed(f"safe_invoke не должен пробрасывать {type(error).__name__}") from error
    _ensure(isinstance(result, tuple) and len(result) == 2, "safe_invoke должен возвращать пару (answer, history)")
    answer, history = result
    _ensure(isinstance(answer, str) and answer, "При превышении лимита safe_invoke возвращает понятный текст")
    _ensure(len(llm.received) == 4, f"При max_steps=4 модель должна быть вызвана 4 раза, вызвана {len(llm.received)}. Подбери recursion_limit")
    _ensure(history == old_history, "При превышении лимита верни исходную историю без незавершённого шага")

    llm = ScriptedChatModel(responses=[AIMessage(content="Привет!")])
    agent = build_agent(llm, tools)
    answer, history = safe_invoke(agent, [], "Привет", max_steps=4)
    _ensure(answer == "Привет!", "В обычном случае safe_invoke возвращает ответ агента")
    _ensure(len(history) == 2 and history[-1].text == "Привет!", "В обычном случае safe_invoke возвращает обновлённую историю, как chat")


def _check_5(extract_trace: Callable, **_):
    messages = [
        HumanMessage(content="Сколько стоит дорога из Москвы в Белград?"),
        AIMessage(content="", tool_calls=[
            {"name": "search_routes", "args": {"origin": "Москва", "destination": "Стамбул"}, "id": "a"},
            {"name": "search_routes", "args": {"origin": "Стамбул", "destination": "Белград"}, "id": "b"},
        ]),
        ToolMessage(content="plane 250 EUR", tool_call_id="a", name="search_routes"),
        ToolMessage(content="Ошибка: сервис недоступен", tool_call_id="b", name="search_routes", status="error"),
        AIMessage(content="Через Стамбул, от 310 EUR."),
    ]
    trace = extract_trace(messages)
    _ensure(isinstance(trace, list) and all(isinstance(x, dict) for x in trace), "extract_trace возвращает список словарей")
    kinds = [x.get("type") for x in trace]
    _ensure(kinds == ["user", "tool_call", "tool_call", "tool_result", "tool_result", "answer"], f"Ожидаемая последовательность type: user, tool_call, tool_call, tool_result, tool_result, answer. Получено: {kinds}")
    _ensure(trace[1].get("name") == "search_routes" and trace[1].get("args") == {"origin": "Москва", "destination": "Стамбул"}, "tool_call: нужны поля name и args")
    _ensure(trace[3].get("content") == "plane 250 EUR" and trace[3].get("name") == "search_routes", "tool_result: нужны поля name и content")
    _ensure(trace[4].get("is_error") is True and trace[3].get("is_error") is False, "tool_result: поле is_error (status == 'error')")
    _ensure(trace[-1].get("content") == "Через Стамбул, от 310 EUR.", "answer: поле content")
    json.dumps(trace, ensure_ascii=False)  # трейс должен сериализоваться


_CHECKS = {
    "1.1": _check_1_1,
    "1.2": _check_1_2,
    "2.1": _check_2_1,
    "2.2": _check_2_2,
    "2.3": _check_2_3,
    "2.4": _check_2_4,
    "2.5": _check_2_5,
    "3": _check_3,
    "4.1": _check_4_1,
    "4.2": _check_4_2,
    "5": _check_5,
}


def check(task: str, **objects) -> None:
    """Запускает тесты задания ``task``. Печатает ✅ или ❌ с причиной."""
    if task not in _CHECKS:
        raise KeyError(f"Нет такого задания: {task}. Есть: {', '.join(_CHECKS)}")
    try:
        _CHECKS[task](**objects)
    except CheckFailed as error:
        print(f"❌ Задание {task}: {error}")
        return
    except TypeError as error:
        if "missing" in str(error) and "required" in str(error):
            print(f"❌ Задание {task}: проверке не передали нужный объект ({error})")
        else:
            print(f"❌ Задание {task}: {type(error).__name__}: {error}")
            traceback.print_exc(limit=3)
        return
    except Exception as error:  # noqa: BLE001
        print(f"❌ Задание {task}: код упал с {type(error).__name__}: {error}")
        traceback.print_exc(limit=3)
        return
    print(f"✅ Задание {task}: базовые тесты пройдены")
