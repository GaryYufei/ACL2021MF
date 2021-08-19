# ACL2021MF
Source Code For ACL 2021 Paper "Mention Flags (MF): Constraining Transformer-based Text Generators"

## Data Download
Please download the evaluation code from [here](https://drive.google.com/drive/folders/10pZHQwNxzTPALzDXqNZokSQOJJtRjB5p?usp=sharing) and put it into the dataset/ folder.

The pre-trained models are available in [here](https://drive.google.com/drive/folders/1pOY_G4ygQ8C76mgGlchyc7jbEtwoY_r9?usp=sharing). Please download each file and put them into the dataset/ folder.

The training, dev and test data for Commonsense Generation and E2E task are available in [here](https://drive.google.com/drive/folders/1i_rua8e3Pl230K9vy3su_wkSZrGykrT2?usp=sharing). Please download each file and put them into the dataset/ folder.

The training, dev and test data for is coming soon.

## Dependency
Before running the code, please install following dependencies:
- python==3.6.1
- transformers==3.5.1
- numpy==1.19.2
- yacs==0.1.6
- tqdm==4.49.0
- torch==1.4.0a0+f067088
- h5py==2.7.0
- anytree==2.7.3
- dataclasses==0.7
- typing==3.6.6

## Running Models

### CommonSen

#### Training all models in the paper

| Model                  | Command                                                                                                                                                                                       |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Trans, L3 Baseline     | python train_T5.py --config t5_base.yml  --config-override use_mention_flag False do_pretrain_lm_init False freeze_param False --serialization-dir dataset/commonGen_transL3_baseline --train |
| Trans, L3 Mention Flag | python train_T5.py --config t5_base.yml  --config-override do_pretrain_lm_init False freeze_param False --serialization-dir dataset/commonGen_transL3_mf --train                              |
| T5-Base Baseline       | python train_T5.py --config t5_base.yml --config-override use_mention_flag False  --serialization-dir dataset/commonGen_t5_base_baseline --train                                              |
| T5-Base Mention Flag   | python train_T5.py --config t5_base.yml  --serialization-dir dataset/commonGen_t5_base_mf --train                                                                                             |
| T5-Large Baseline      | python train_T5.py --config t5_large.yml --config-override use_mention_flag False  --serialization-dir dataset/commonGen_t5_large_baseline --train                                            |
| T5-Large Mention Flag  | python train_T5.py --config t5_large.yml  --serialization-dir dataset/commonGen_t5_large_mf --train                                                                                           |
| T5-Base Scalar Mf      | python train_T5.py --config t5_base.yml --config-override use_mf_scalar True --serialization-dir dataset/commonGen_t5_base_scalar_mf --train                                                  |
| T5-Base Static Mf      | python train_T5.py --config t5_base.yml --config-override static_mf True --serialization-dir dataset/commonGen_t5_base_static_mf --train                                                      |

#### Evluating models

|         Model         |                                                                                              Command                                                                                             |
|:---------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  T5-Base Mention Flag |  python train_T5.py --config dataset/commonGen_t5_base_mf/config.yml  --start-from-checkpoint dataset/commonGen_t5_base_mf  --test --seen-constraint-path dataset/commonGen_seen_constraint.txt  |
| T5-Large Mention Flag | python train_T5.py --config dataset/commonGen_t5_large_mf/config.yml  --start-from-checkpoint dataset/commonGen_t5_large_mf  --test --seen-constraint-path dataset/commonGen_seen_constraint.txt |

### E2E

#### Training all models in the paper

|          Model         |                                                                                         Command                                                                                         |
|:----------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|    T5-Base Baseline    |                        python train_e2e_T5.py --config e2e_t5_base.yml --serialization-dir dataset/e2e_baseline --train  --config-override use_mention_flag False                       |
|  T5-Base Mention Flag  |                                                python train_e2e_T5.py --config e2e_t5_base.yml --serialization-dir dataset/e2e_mf --train                                               |
|   Trans, L3 Baseline   | python train_e2e_T5.py --config e2e_t5_base.yml --serialization-dir dataset/e2e_transL3_baseline --train  --config-override use_mention_flag False do_pretrain_lm_init False freeze_param False |
| Trans, L3 Mention Flag |             python train_e2e_T5.py --config e2e_t5_base.yml --serialization-dir dataset/e2e_transL3_mf --train  --config-override do_pretrain_lm_init False freeze_param False            |
|    T5-Base Static MF   |                               python train_e2e_T5.py --config e2e_t5_base.yml  --serialization-dir dataset/e2e_static_mf --train --config-override static_mf True                              |
|    T5-Base Scalar MF   |                             python train_e2e_T5.py --config e2e_t5_base.yml  --serialization-dir dataset/e2e_scalar_mf --train --config-override use_mf_scalar True                            |
|    T5-Base Merged MF   |                            python train_e2e_T5.py --config e2e_t5_base.yml   --serialization-dir dataset/e2e_merged_mf --train --config-override use_mf_merged True                            |

#### Evaluating models

|  Model  |                                                 Command                                                 |
|:-------:|:-------------------------------------------------------------------------------------------------------:|
| T5-Base | python train_e2e_T5.py --config dataset/e2e_mf/config.yml --start-from-checkpoint dataset/e2e_mf --test |


### nocaps

#### Training all models in the paper

|          Model         |                                                                                                   Command                                                                                                   |
|:----------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|    T5-Base Baseline    |                            python train_COCO_T5.py --config COCO_t5_base_nocaps.yml  --serialization-dir dataset/nocaps_baseline --train --config-override use_mention_flag False                           |
|  T5-Base Mention Flags |                                                   python train_COCO_T5.py --config COCO_t5_base_nocaps.yml  --serialization-dir dataset/nocaps_mf --train                                                   |
|    Trans L3 Baseline   | python train_COCO_T5.py --config COCO_t5_base_nocaps.yml  --serialization-dir dataset/nocaps_baseline_transL3 --train --config-override use_mention_flag False do_pretrain_lm_init False freeze_param False |
| Trans L3 Mention Flags |     python train_COCO_T5.py --config COCO_t5_base_nocaps.yml  --serialization-dir dataset/nocaps_mf_transL3 --train --config-override use_mention_flag True do_pretrain_lm_init False freeze_param False    |
|    T5-Base Scalar MF   |                             python train_COCO_T5.py --config COCO_t5_base_nocaps.yml  --serialization-dir dataset/nocaps_scalar_mf --train --config-override use_mf_scalar True                             |
|    T5-Base Static MF   |                               python train_COCO_T5.py --config COCO_t5_base_nocaps.yml  --serialization-dir dataset/nocaps_static_mf --train --config-override static_mf True                               |

#### Evaluating models

|         Model        |                                                                                       Command                                                                                      |
|:--------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| T5-Base Mention Flag | python train_COCO_T5.py --config dataset/nocaps_mf/config.yml  --start-from-checkpoint dataset/nocaps_mf --validation  --novel-constraint-path dataset/nocaps_novel_constraint.txt |
