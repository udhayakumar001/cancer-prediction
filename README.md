# Cancer Prediction

This project explores lung cancer risk levels with data visualizations and three classifiers: Random Forest, XGBoost, and an MLP neural network.

## Run

From this folder, run:

```powershell
$env:MPLBACKEND='Agg'
python -u .\cancer_predection.py
```

The noninteractive Matplotlib backend lets the script complete without opening plot windows. The dataset is read from the same folder as the Python script.

## Latest run

The program completed successfully with exit code 0. It used 1,000 dataset rows, split into 800 training and 200 test rows.

| Model | Test accuracy | Test F1 |
| --- | ---: | ---: |
| Random Forest | 98% | 98% |
| XGBoost | 94% | 94% |
| MLP | Tuned; metrics are not reported by the current script | — |

The screenshot below shows the model metrics printed during the run. Full console output is in [`run_output.txt`](run_output.txt).

![Screenshot of the program output](run_output_screenshot.png)

## Visualizations

The supplied plots are stored in [`figures/`](figures/).

### Age and feature distributions

![Figure 1: Age distribution box plot](figures/Figure_1.png)

![Figure 2: Age normal quantile plot](figures/Figure_2.png)

![Figure 12: Feature distributions](figures/Figure_12.png)

![Figure 13: Gender distribution by cancer level](figures/Figure_13.png)

![Figure 14: Age distribution by cancer level](figures/Figure_14.png)

### Correlations and cancer-level balance

![Figure 3: Annotated correlation matrix](figures/Figure_3.png)

![Figure 4: Correlation matrix](figures/Figure_4.png)

![Figure 5: Correlation heatmap](figures/Figure_5.png)

![Figure 10: Cancer-level counts](figures/Figure_10.png)

![Figure 11: Cancer-level proportions](figures/Figure_11.png)

### Model results

![Figure 6: Random Forest confusion matrix](figures/Figure_6.png)

![Figure 7: XGBoost confusion matrix](figures/Figure_7.png)

![Figure 8: Model training and test F1 scores](figures/Figure_8.png)

![Figure 9: XGBoost confusion matrix](figures/Figure_9.png)

![Figure 15: Random Forest feature importance](figures/Figure_15.png)
