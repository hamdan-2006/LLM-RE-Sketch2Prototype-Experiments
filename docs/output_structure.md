# Output Structure

Each output folder should follow this pattern:

```text
experiments_output/
└── experiment_log.txt
└── experiment_manifest.csv
    └── SCENARIO_NAME/
        ├── requirements_run1.txt
        ├── requirements_run2.txt
        ├── requirements_run3.txt
        ├── prototype_raw_run1.txt
        ├── prototype_raw_run2.txt
        ├── prototype_raw_run3.txt
        ├── prototype_run1.html
        ├── prototype_run2.html
        ├── prototype_run3.html
        
```

## Recommended Manifest Columns

```text
model,scenario,run,requirements_file,prototype_raw_file,prototype_html_file,success,truncated,error,date
```
