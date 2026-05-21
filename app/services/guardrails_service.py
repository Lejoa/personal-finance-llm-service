"""
GuardrailsService — Capa de validación de input/output para el LLM financiero.

Valida que:
- El input del usuario sea una pregunta financiera (RestrictToTopic) — usado por /llm/financial-insights
- El input no contenga lenguaje tóxico (ToxicLanguage)
- El input no exponga datos personales sensibles (DetectPII)
- El output del LLM no contenga lenguaje tóxico (ToxicLanguage)

Para /llm/chat se usa validate_safety() que omite RestrictToTopic:
la detección de mensajes off_topic es responsabilidad del LLM (intent="off_topic").

Nota: Guardrails no tiene API async nativa. Los métodos async_* usan
run_in_executor para correr la validación síncrona en el threadpool de asyncio,
evitando el warning "Could not obtain an event loop" y sin bloquear el event loop.
"""
import asyncio

# GUARDRAILS HUB VALIDATORS — TEMPORARILY DISABLED (2026-05-18)
# RestrictToTopic, ToxicLanguage and DetectPII cannot be installed because
# the Guardrails Hub registry removed them from PyPI (supply chain incident 2026-05-11).
# All validation methods below return without error until the Hub is restored.
# To re-enable: uncomment these imports and remove the pass statements in each method.
#
# from guardrails import Guard
# from guardrails.hub import RestrictToTopic, ToxicLanguage, DetectPII


FINANCIAL_TOPICS = [
    "personal finance",
    "savings",
    "investment",
    "budget",
    "expenses",
    "income",
    "debt",
    "loans",
    "financial education",
    "cryptocurrency",
    "stock market",
    "money management",
    # Registro de gastos e ingresos en lenguaje natural
    "spending money",
    "buying something",
    "purchasing goods",
    "paying for something",
    "expense tracking",
    "recording a transaction",
    "logging an expense",
    "logging income",
    "received payment",
    "earned money",
    "spent money",
    "paid for",
    "transaction record",
    "financial transaction",
    "purchase amount",
    "money spent",
    "money received",
]

INVALID_TOPICS = [
    "cooking recipes",
    "sports scores",
    "politics",
    "software programming",
    "entertainment news",
]

PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "CO_NIT",
]


class GuardrailsValidationError(Exception):
    def __init__(self, message: str, error_type: str):
        self.message = message
        self.error_type = error_type
        super().__init__(message)


class GuardrailsService:
    def __init__(self):
        # TEMPORARILY DISABLED — re-enable when Hub validators are available again.
        # Uncomment the blocks below and remove the pass statements in each method.
        #
        # self.input_guard = Guard().use_many(
        #     RestrictToTopic(
        #         valid_topics=FINANCIAL_TOPICS,
        #         invalid_topics=INVALID_TOPICS,
        #         disable_classifier=False,
        #         disable_llm=True,
        #         on_fail="exception",
        #     ),
        #     ToxicLanguage(
        #         threshold=0.5,
        #         validation_method="sentence",
        #         on_fail="exception",
        #     ),
        #     DetectPII(
        #         pii_entities=PII_ENTITIES,
        #         on_fail="exception",
        #     ),
        # )
        # self.safety_guard = Guard().use_many(
        #     ToxicLanguage(
        #         threshold=0.5,
        #         validation_method="sentence",
        #         on_fail="exception",
        #     ),
        #     DetectPII(
        #         pii_entities=PII_ENTITIES,
        #         on_fail="exception",
        #     ),
        # )
        # self.output_guard = Guard().use(
        #     ToxicLanguage(
        #         threshold=0.5,
        #         validation_method="sentence",
        #         on_fail="exception",
        #     ),
        # )
        pass

    def validate_input(self, user_message: str) -> None:
        """
        Valida el mensaje del usuario antes de enviarlo al LLM.
        Lanza GuardrailsValidationError si falla alguna validación.
        """
        # TEMPORARILY DISABLED — validators unavailable (Hub registry down 2026-05-11).
        # Re-enable by uncommenting the block below and removing the pass.
        #
        # try:
        #     self.input_guard.validate(user_message)
        # except Exception as e:
        #     error_msg = str(e).lower()
        #     if "restricttotopic" in error_msg or "topic" in error_msg:
        #         raise GuardrailsValidationError(
        #             "Lo siento, solo puedo ayudarte con preguntas sobre finanzas personales. "
        #             "¿Tienes alguna consulta sobre tu presupuesto, ahorro o inversiones?",
        #             error_type="off_topic",
        #         )
        #     if "toxic" in error_msg:
        #         raise GuardrailsValidationError(
        #             "Tu mensaje contiene contenido inapropiado. "
        #             "Por favor, reformula tu pregunta de forma respetuosa.",
        #             error_type="toxic",
        #         )
        #     if "pii" in error_msg:
        #         raise GuardrailsValidationError(
        #             "Tu mensaje parece contener información personal sensible "
        #             "(como número de cédula o cuenta). "
        #             "Por favor, no incluyas datos personales en tus consultas.",
        #             error_type="pii",
        #         )
        #     raise GuardrailsValidationError(
        #         "No se pudo procesar tu mensaje. Por favor intenta de nuevo.",
        #         error_type="unknown",
        #     )
        pass

    def validate_safety(self, user_message: str) -> None:
        """
        Valida ToxicLanguage y DetectPII únicamente (versión síncrona).
        """
        # TEMPORARILY DISABLED — re-enable by uncommenting the block below.
        #
        # try:
        #     self.safety_guard.validate(user_message)
        # except Exception as e:
        #     error_msg = str(e).lower()
        #     if "toxic" in error_msg:
        #         raise GuardrailsValidationError(
        #             "Tu mensaje contiene contenido inapropiado. "
        #             "Por favor, reformula tu pregunta de forma respetuosa.",
        #             error_type="toxic",
        #         )
        #     if "pii" in error_msg:
        #         raise GuardrailsValidationError(
        #             "Tu mensaje parece contener información personal sensible "
        #             "(como número de cédula o cuenta). "
        #             "Por favor, no incluyas datos personales en tus consultas.",
        #             error_type="pii",
        #         )
        #     raise GuardrailsValidationError(
        #         "No se pudo procesar tu mensaje. Por favor intenta de nuevo.",
        #         error_type="unknown",
        #     )
        pass

    async def async_validate_safety(self, user_message: str) -> None:
        """
        Versión async de validate_safety: corre la validación síncrona en el
        threadpool de asyncio para evitar bloquear el event loop.

        Nota: Guardrails v0.6.8 llama internamente a get_loop() que lanza RuntimeError
        cuando detecta un event loop activo, cayendo al modo síncrono (SequentialValidatorService).
        Esto es el comportamiento esperado — la advertencia en logs es informativa, no un error.
        El run_in_executor es suficiente para no bloquear el event loop de FastAPI.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.validate_safety, user_message)

    def validate_output(self, llm_response: str) -> str:
        """
        Valida la respuesta del LLM antes de retornarla al usuario (versión síncrona).
        """
        # TEMPORARILY DISABLED — re-enable by uncommenting the block below.
        #
        # try:
        #     result = self.output_guard.validate(llm_response)
        #     return result.validated_output or llm_response
        # except Exception:
        #     raise GuardrailsValidationError(
        #         "La respuesta generada no pudo ser validada. Por favor intenta de nuevo.",
        #         error_type="output_invalid",
        #     )
        return llm_response

    async def async_validate_output(self, llm_response: str) -> str:
        """
        Versión async de validate_output: corre la validación síncrona en el
        threadpool de asyncio para evitar bloquear el event loop.

        Nota: igual que async_validate_safety, Guardrails cae al modo síncrono por diseño.
        La advertencia en logs es esperada en v0.6.8; la validación funciona correctamente.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.validate_output, llm_response)


# Singleton — se instancia una vez al inicio del servicio
_guardrails_service: GuardrailsService | None = None


def get_guardrails_service() -> GuardrailsService:
    global _guardrails_service
    if _guardrails_service is None:
        _guardrails_service = GuardrailsService()
    return _guardrails_service
