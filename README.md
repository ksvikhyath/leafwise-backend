# LeafWise

LeafWise is an explainable AI system for medicinal plant identification and farmer advisory generation.

It combines:

- BEiT-based image classification
- TF-IDF + logistic regression metadata classification
- advisory generation for medicinal uses, cultivation, and income potential
- translation support for regional language output
- explainability tools such as Grad-CAM, LIME, LRP, ELI5, SHAP, and Anchor

## Naming Convention

The image folders are the canonical labels and follow the schema:

`Common Name (Scientific Name)`

Examples:

- `Aloe vera (Aloe vera)`
- `Amla (Phyllanthus emblica)`
- `ashoka (Saraca asoca)`

The JSON knowledge base and metadata loading code normalize older aliases to these folder names.

## Workspace Layout

```text
LeafWise_optimised/
├── advisory.py
├── beit v3.py
├── camv2.py
├── explainers.py
├── lime_image_explainer.py
├── lrp_image_explainer.py
├── metadata_text_trainer.py
├── model_utils.py
├── translator.py
├── xai_dashboard.py
├── MetaData_Cleaned.csv
└── plant_knowledge.json

Indian Medicinal Leaves/
├── Aloe vera (Aloe vera)/
├── Amla (Phyllanthus emblica)/
├── ashoka (Saraca asoca)/
└── ...
```

## Run

1. Install dependencies.
2. Train the text model:
   - `python metadata_text_trainer.py`
    - Optional Optuna tuning for metadata model:
       - PowerShell:
          - `$env:LEAFWISE_ENABLE_OPTUNA='1'; $env:LEAFWISE_TEXT_OPTUNA_TRIALS='30'; python metadata_text_trainer.py`
       - Best trial output is saved in `checkpoints/metadata_optuna_best_params.json`.
3. Train the image model:
   - `python "beit v3.py"`
   - Optional Optuna tuning before final training:
     - PowerShell:
       - `$env:LEAFWISE_ENABLE_OPTUNA='1'; $env:LEAFWISE_OPTUNA_TRIALS='20'; python "beit v3.py"`
     - Best trial parameters are saved in `checkpoints/optuna_best_params.json`.
4. Launch the dashboard:
   - `streamlit run xai_dashboard.py`

## Notes

- Paths are resolved relative to the workspace.
- `advisory.py` accepts canonical folder labels and older aliases.
- The dashboard title is now LeafWise.

## Credits

- Naveen S
- Hrudhay R
- Vikhyath