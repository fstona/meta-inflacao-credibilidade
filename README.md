# Meta de inflação, credibilidade e juros

Subir a meta de inflação permitiria ao Banco Central cortar juros?

Com as expectativas de inflação acima da meta, volta ao debate a ideia de que elevar a meta de 3% para algo próximo de 4,5% permitiria ao Banco Central reduzir os juros. Este estudo avalia esse argumento de forma quantitativa. Construímos um modelo estilizado em que as expectativas dependem da credibilidade do BC e da tolerância percebida com desvios da meta. O modelo identifica em que condições uma meta mais alta poderia reduzir a Selic. Em seguida, estimamos os parâmetros com dados do Focus e com o histórico de decisões do CMN, incluindo o debate sobre a meta em 2023.

## Principais resultados

- Na teoria, subir a meta só reduz a Selic se três condições valerem ao mesmo tempo: credibilidade baixa, expectativas presas a um nível que não depende da meta e nenhuma perda de credibilidade com a mudança.
- Nas seis mudanças de meta entre 2017 e 2021, as expectativas acompanharam o novo alvo quase um para um (repasse de 0,98; ω = 0,94).
- A desancoragem passada reduz a credibilidade de forma significativa (η = 0,37, t ≈ 3,6).
- O debate de 2023 implica uma perda de credibilidade com a mudança (γ) entre 0,10 e 0,17.
- Com os parâmetros estimados, elevar hoje a meta de 3% para 4,5% aumentaria a Selic neutra nominal em cerca de 2,5 p.p. no cenário central (de +1,4 a +3,2 p.p.).

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `nota_tecnica/` | Nota técnica em LaTeX (`meta_inflacao.tex`), PDF compilado e figuras |
| `modelo_estilizado/` | Simulações do modelo com calibração ilustrativa e mapas de sensibilidade |
| `estimacao/` | Dados, scripts de estimação (trimestral, horizonte constante de 36 meses), calibração de γ e resultados |

## Como rodar

Requer Python 3 com `numpy`, `pandas` e `matplotlib`.

```bash
cd estimacao
python calibra_trimestral.py   # λ, η, ω -> base_trimestral.csv, eventos_trim.csv, parametros_trim.csv
python modelo_trim.py          # simulação da meta 3% -> 4,5% (bases 2026T3 e 2026T4)
python gamma_2023.py           # calibração de γ pelo episódio de 2023
python graficos_trim.py        # figuras da estimação
python grafico_2023.py         # figura do Focus diário em 2023

cd ../modelo_estilizado
python toy_desancoragem.py
python toy_lambda_endogeno.py
python mapa_kappa_omega_lambda.py
python equilibrio_meta.py
```

`calibra_trimestral.py` reaproveita a leitura de dados e a reconstrução das metas de `calibra.py`.

Para compilar a nota: `cd nota_tecnica && pdflatex meta_inflacao.tex` (duas vezes, para as referências).

## Dados

- `focus.txt`: medianas do Focus/BCB para o IPCA (12 meses à frente e anos-calendário t+1 a t+4), média mensal das observações diárias, dez/2001 a nov/2026 (novembro parcial). Formato: `aaaamm,12m,t+1,t+2,t+3,t+4`.
- `ipca.txt`: IPCA acumulado em 12 meses, SGS 13522. Formato: `aaaamm,valor`.
- `focus_2023_diario.txt`: medianas diárias do Focus para o IPCA de 2025, 2026 e 2027, nov/2022 a dez/2023 (dias com mudança e um a cada cinco dias úteis).
- Metas do CMN: reconstruídas no código a partir das datas dos anúncios.

Fontes: Banco Central do Brasil (Sistema de Expectativas de Mercado e SGS) e resoluções do Conselho Monetário Nacional.

## Ressalvas

Modelo ilustrativo em estado estacionário. θ e r\* são calibrados, não estimados. ω vem de seis eventos de redução da meta, e γ de um único episódio de ameaça de aumento. Os números mostram o mecanismo e não são projeções. Detalhes na seção de limitações da nota técnica.
