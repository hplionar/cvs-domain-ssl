# CVS Experiment Results Summary

| experiment_id | dataset | loss | epochs_completed | best_map_epoch | best_map | best_map_mean_auc | best_map_mean_bacc | best_bacc_epoch | best_bacc_mean_bacc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001_dinov2_b_frame_frozen | endoscapes | normal_bce | 5 | 2 | 0.4834 | 0.8179 | 0.5712 | 5 | 0.5976 |
| exp002_dinov2_b_frame_frozen_weighted | endoscapes | weighted_bce | 5 | 1 | 0.4771 | 0.8104 | 0.7157 | 5 | 0.7226 |
| exp003_dinov2_b_sages_frame_frozen | sages | normal_bce | 5 | 4 | 0.4262 | 0.7736 | 0.6373 | 4 | 0.6373 |
| exp004_dinov2_b_sages_frame_frozen_weighted | sages | weighted_bce | 5 | 4 | 0.4304 | 0.7841 | 0.7102 | 4 | 0.7102 |
| smoke_dinov2_b | endoscapes | normal_bce | 1 |  |  |  |  |  |  |
| smoke_dinov2_b_more_val | endoscapes | normal_bce | 1 |  |  |  |  |  |  |
| smoke_dinov2_b_weighted | endoscapes | weighted_bce | 1 |  |  |  |  |  |  |
| smoke_sages_clip_meanpool | sages | normal_bce | 1 |  |  |  |  |  |  |
| smoke_sages_dinov2_b_normal | sages | normal_bce | 1 | 1 | 0.2033 | 0.5791 | 0.5226 | 1 | 0.5226 |
| smoke_sages_dinov2_b_weighted | sages | weighted_bce | 1 | 1 | 0.2002 | 0.5740 | 0.5330 | 1 | 0.5330 |
