"""
GuardrailsService — Capa de validación de input/output para el LLM financiero.

Validators activos:
  ToxicLanguage  — rechaza lenguaje ofensivo o amenazante (inglés y español,
                   modelo multilingual XLMRoberta).
  DetectPII      — rechaza mensajes que expongan email, teléfono o tarjeta.
  ToxicLanguage  — valida que la respuesta del LLM no sea tóxica (output_guard).

Uso por endpoint:
  /llm/chat              → async_validate_safety()  (ToxicLanguage + DetectPII)
                           async_validate_output()  (ToxicLanguage en respuesta)
                           La detección de mensajes off_topic es responsabilidad
                           del LLM mediante intent="off_topic" en el chat chain.
                           RestrictToTopic no se usa porque bloquearía mensajes
                           de registro de transacciones en español.

  /llm/financial-insights → sin validación Guardrails.
                           El campo `goal` proviene de un formulario financiero
                           controlado; el contexto de uso garantiza que el input
                           es financiero sin necesidad de validators adicionales.

Nota: Guardrails no tiene API async nativa. Los métodos async_* usan
run_in_executor para correr la validación síncrona en el threadpool de asyncio,
evitando el warning "Could not obtain an event loop" y sin bloquear el event loop.
"""
import asyncio

from guardrails import Guard
from guardrails.hub import ToxicLanguage, DetectPII


PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    # CO_NIT removido: Presidio carga solo recognizers en inglés por defecto;
    # el recognizer CO_NIT (español) no se registra y no detecta nada.
]


class GuardrailsValidationError(Exception):
    def __init__(self, message: str, error_type: str):
        self.message = message
        self.error_type = error_type
        super().__init__(message)


class GuardrailsService:
    def __init__(self):
        self.safety_guard = Guard().use_many(
            ToxicLanguage(
                threshold=0.5,
                validation_method="sentence",
                model_name="multilingual",
                on_fail="exception",
            ),
            DetectPII(
                pii_entities=PII_ENTITIES,
                on_fail="exception",
            ),
        )
        self.output_guard = Guard().use(
            ToxicLanguage(
                threshold=0.5,
                validation_method="sentence",
                model_name="multilingual",
                on_fail="exception",
            ),
        )

    def validate_safety(self, user_message: str) -> None:
        """
        Valida ToxicLanguage y DetectPII. Base síncrona de async_validate_safety.
        """
        try:
            self.safety_guard.validate(user_message)
        except Exception as e:
            error_msg = str(e).lower()
            if "toxic" in error_msg:
                raise GuardrailsValidationError(
                    "Tu mensaje contiene contenido inapropiado. "
                    "Por favor, reformula tu pregunta de forma respetuosa.",
                    error_type="toxic",
                )
            if "pii" in error_msg:
                raise GuardrailsValidationError(
                    "Tu mensaje parece contener información personal sensible "
                    "(como número de cédula o cuenta). "
                    "Por favor, no incluyas datos personales en tus consultas.",
                    error_type="pii",
                )
            raise GuardrailsValidationError(
                "No se pudo procesar tu mensaje. Por favor intenta de nuevo.",
                error_type="unknown",
            )

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
        try:
            result = self.output_guard.validate(llm_response)
            return result.validated_output or llm_response
        except Exception:
            raise GuardrailsValidationError(
                "La respuesta generada no pudo ser validada. Por favor intenta de nuevo.",
                error_type="output_invalid",
            )

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
