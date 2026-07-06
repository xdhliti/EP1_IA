# EP1 IA - Arvores de Decisao e Florestas Obliquas

Implementacoes em Python/NumPy para o EP1 de Inteligencia Artificial.

## Arquivos principais

- `arvore_decisao_ortogonal.py`: arvore de decisao ortogonal do zero.
- `random_forest_ortogonal.py`: random forest ortogonal do zero.
- `arvore_decisao_obliqua.py`: arvore de decisao obliqua com hiperplanos locais.
- `random_forest_obliqua.py`: oblique random forest com bootstrap e voto majoritario.

## Dados

Os scripts esperam um arquivo local chamado `data.npz` na raiz do projeto, contendo:

- `X_train`
- `y_train`
- `X_test`

O arquivo de dados e as predicoes geradas nao sao versionados no Git por padrao.

## Como executar

```bash
python arvore_decisao_ortogonal.py
python random_forest_ortogonal.py
python arvore_decisao_obliqua.py
python random_forest_obliqua.py
```

Cada script imprime acuracia de validacao, tempo de treino, tempo de predicao e matriz de confusao.

## Melhor resultado local ate agora

O resultado de validacao observado com oRF foi `random_forest_obliqua.py`:

```text
Acuracia de validacao: 0.7319
```

Essa configuracao prioriza acuracia, com custo de treino mais alto.
