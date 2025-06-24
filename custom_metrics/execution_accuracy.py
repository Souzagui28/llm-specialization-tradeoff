import sqlite3
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class ExecutionAccuracy(BaseMetric):
    """
    Métrica customizada para DeepEval que mede a acurácia de execução de uma consulta SQL.
    """
    def __init__(self, db_path: str, threshold: float = 1.0):
        self.threshold = threshold
        self.db_path = db_path

    def _execute_query(self, query: str):
        """Executa uma consulta de forma segura e retorna os resultados."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
                # Para garantir consistência na comparação, ordenamos os resultados
                # Embora a comparação final seja com sets, isso ajuda a padronizar.
                return sorted(results) 
        except Exception as e:
            # Se a query falhar (erro de sintaxe, etc.), retorna um identificador de erro
            return f"Execution Error: {e}"

    def measure(self, test_case: LLMTestCase) -> float:
        # A lógica interna do measure, conforme o requisito 3.1
        generated_sql = test_case.actual_output
        ground_truth_sql = test_case.expected_output

        # b. Executar a consulta SQL gerada
        generated_results = self._execute_query(generated_sql)
        # c. Executar a consulta ground truth
        ground_truth_results = self._execute_query(ground_truth_sql)

        # Se houver erro na execução de qualquer uma das queries, consideramos falha
        if isinstance(generated_results, str) or isinstance(ground_truth_results, str):
            self.score = 0.0
            self.reason = f"Falha na execução. Gerado: {generated_results}, Esperado: {ground_truth_results}"
            return self.score

        # d. Comparar os conjuntos de resultados de forma insensível à ordem
        # Converter a lista de tuplas para um set de tuplas é a forma mais eficaz de fazer isso.
        success = set(generated_results) == set(ground_truth_results)
        
        # e. Retornar 1.0 para sucesso e 0.0 para falha
        self.score = 1.0 if success else 0.0
        self.reason = "Resultados idênticos" if success else "Resultados divergentes"
        
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Execution Accuracy"