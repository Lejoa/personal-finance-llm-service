"""
GuardrailsService — Capa de validación de input/output para el LLM financiero.

Valida que:
- El input del usuario sea una pregunta financiera (RestrictToTopic)
- El input no contenga lenguaje tóxico (ToxicLanguage)
- El input no exponga datos personales sensibles (DetectPII)
- El output del LLM no contenga lenguaje tóxico (ToxicLanguage)
"""
from guardrails import Guard
from guardrails.hub import RestrictToTopic, ToxicLanguage, DetectPII


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
]

INVALID_TOPICS = [
    "cooking recipes",
    "sports",
    "politics",
    "programming",
    "entertainment",
    "travel",
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
        # Guard para INPUT del usuario
        self.input_guard = Guard().use_many(
            RestrictToTopic(
                valid_topics=FINANCIAL_TOPICS,
                invalid_topics=INVALID_TOPICS,
                disable_classifier=False,
                disable_llm=True,
                on_fail="exception",
            ),
            ToxicLanguage(
                threshold=0.5,
                validation_method="sentence",
                on_fail="exception",
            ),
            DetectPII(
                pii_entities=PII_ENTITIES,
                on_fail="exception",
            ),
        )

        # Guard para OUTPUT del LLM
        self.output_guard = Guard().use(
            ToxicLanguage(
                threshold=0.5,
                validation_method="sentence",
                on_fail="exception",
            ),
        )

    def validate_input(self, user_message: str) -> None:
        """
        Valida el mensaje del usuario antes de enviarlo al LLM.
        Lanza GuardrailsValidationError si falla alguna validación.
        """
        try:
            self.input_guard.validate(user_message)
        except Exception as e:
            error_msg = str(e).lower()
            if "restricttotopic" in error_msg or "topic" in error_msg:
                raise GuardrailsValidationError(
                    "Lo siento, solo puedo ayudarte con preguntas sobre finanzas personales. "
                    "¿Tienes alguna consulta sobre tu presupuesto, ahorro o inversiones?",
                    error_type="off_topic",
                )
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

    def validate_output(self, llm_response: str) -> str:
        """
        Valida la respuesta del LLM antes de retornarla al usuario.
        Retorna la respuesta si es válida.
        Lanza GuardrailsValidationError si el output es inapropiado.
        """
        try:
            result = self.output_guard.validate(llm_response)
            return result.validated_output or llm_response
        except Exception:
            raise GuardrailsValidationError(
                "La respuesta generada no pudo ser validada. Por favor intenta de nuevo.",
                error_type="output_invalid",
            )


# Singleton — se instancia una vez al inicio del servicio
_guardrails_service: GuardrailsService | None = None


def get_guardrails_service() -> GuardrailsService:
    global _guardrails_service
    if _guardrails_service is None:
        _guardrails_service = GuardrailsService()
    return _guardrails_service
