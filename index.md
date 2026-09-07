# Sleep Telemetry Analysis: Can Pre-Bed Workouts Coexist with Restorative Sleep?

**Author**: Christopher Nguyen  
**Project**: DSC 80 Final Project (UC San Diego)  
**Live Site**: [https://ccnguyen106.github.io/sleep-telemetry-analysis/](https://ccnguyen106.github.io/sleep-telemetry-analysis/)

---

## Introduction

Standard clinical sleep hygiene guidelines typically caution against exercising in the hours directly leading up to bedtime. The theory is intuitive: rigorous physical exertion elevates heart rate, boosts core body temperature, and triggers sympathetic nervous system arousal, delaying sleep onset. However, for university students and working professionals, late-evening hours are often the only opportunity for structured workouts, fitness classes, or outdoor running. Furthermore, emerging wearable research suggests light-to-moderate movement may act as a stress buffer, lowering cortisol and facilitating sleep onset.

This project investigates the real-world interplay between pre-bedtime physical movement and subsequent sleep dynamics using the **UCSD ExtraSensory Dataset**:

> **Primary Research Question**: *Does physical activity in the two hours leading up to bedtime affect nocturnal sleep duration and sleep restlessness?*

### The ExtraSensory Dataset
The raw ExtraSensory dataset records minute-by-minute sensor readings and self-reported behavioral logs from 60 unique participants in naturalistic, free-living environments, totaling **377,346 rows across 279 columns**.

To prevent computational bottlenecks while isolating the variables relevant to bedtime behavior, I subsetted the dataset down to 14 core columns capturing physical movement, sleep state, phone dynamics, screen activity, and charging status:

| Column Name | Data Type | Description | Why I Kept It |
| :--- | :--- | :--- | :--- |
| `uuid` | string | Unique 36-character participant ID | Needed to track each person separately so I don't accidentally mix up data between users. |
| `timestamp` | int64 | Unix epoch timestamp in seconds | Lets me sort readings chronologically and calculate exactly how long each sleep session lasted. |
| `label:FIX_walking` | float64 | 1.0 if walking, NaN otherwise | Tracks casual walking around the house or outside in the 2 hours before bed. |
| `label:FIX_running` | float64 | 1.0 if running, NaN otherwise | Flags intense cardio or late-night runs right before sleep. |
| `label:OR_exercise` | float64 | 1.0 if exercising, NaN otherwise | Catches general workouts or gym sessions that walking and running tags might miss. |
| `label:SLEEPING` | float64 | 1.0 if sleeping, 0.0 if awake | The ground-truth label I use to define when someone actually fell asleep and woke up. |
| `raw_acc:magnitude_stats:mean` | float64 | Mean acceleration magnitude of phone | Gives a baseline check for whether the phone was being moved around or sitting still. |
| `raw_acc:magnitude_stats:std` | float64 | Standard deviation of phone acceleration | My main proxy for sleep restlessness—more acceleration spread means more tossing and turning. |
| `watch_acceleration:magnitude_stats:mean` | float64 | Mean magnitude of smartwatch acceleration | Checks overall wrist movement from smartwatch telemetry. |
| `watch_acceleration:magnitude_stats:std` | float64 | Standard deviation of smartwatch acceleration | A backup proxy for sleep restlessness based on wrist movement rather than phone movement. |
| `lf_measurements:screen_brightness` | float64 | Screen brightness reading (0.0 to 1.0) | Context on whether the user was scrolling on their phone before bed; also used in my missingness tests. |
| `lf_measurements:light` | float64 | Ambient light sensor measurement | Helps check whether the lights were actually turned off when they tried to sleep. |
| `discrete:app_state:is_active` | float64 | 1.0 if app was in foreground | Tells me if the user was actively using the app versus letting it run in the background. |
| `discrete:battery_state:is_charging` | float64 | 1.0 if plugged into power, 0.0 otherwise | Checks if the phone was plugged in on a nightstand, which is central to my missingness analysis. |

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning Process & Session Extraction
The raw ExtraSensory telemetry is structured as 1-minute time slices. To analyze sleep sessions rather than isolated minutes, I performed the following data cleaning and aggregation steps:

1. **Datetime Conversion**: Converted Unix epoch seconds (`timestamp`) into pandas `datetime64` objects (`date_time`) to support rolling window lookups and calendar extraction.
2. **Activity Label Imputation**: In the ExtraSensory data collection app, participants only checked boxes for behaviors they were actively performing. An unselected checkmark appears as `NaN` in raw CSVs, representing the absence of activity rather than corrupted telemetry. I imputed `NaN` values in `label:FIX_walking`, `label:FIX_running`, and `label:OR_exercise` with `0.0`.
3. **Compound Activity Indicator**: Created an `is_active` binary indicator equal to `1` if a participant was walking, running, or exercising during that minute, and `0` otherwise.
4. **Sleep Session Aggregation**: Identified continuous blocks where `label:SLEEPING == 1.0` using cumulative sums over user-level state changes.
5. **Session Filtering**: Dropped sleep blocks shorter than 2 hours (120 minutes) or longer than 14 hours (840 minutes). This removed brief daytime naps, logging omissions, and multi-day sensor artifacts, isolating 224 clean nocturnal sleep episodes.
6. **2-Hour Pre-Bed Lookback Window**: For each valid sleep episode, I queried the 120-minute window directly preceding `sleep_start`. I computed total active minutes (`pre_bed_active_mins`), defined a workout indicator (`exercised_before_bed = 1` for $\ge 15$ active minutes), and extracted calendar attributes (`start_hour`, `day_of_week`, `is_weekend`).

### Cleaned Dataset Preview
Below are the first 5 rows of the aggregated sleep sessions DataFrame:

| uuid | sleep_start | sleep_end | duration_mins | pre_bed_active_mins | exercised_before_bed | restlessness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `00EABED2-271D-49D8-B599-1D4A09240601` | 2015-10-08 18:02:33 | 2015-10-08 21:50:46 | 228.22 | 0 | 0 | 0.0605 |
| `00EABED2-271D-49D8-B599-1D4A09240601` | 2015-10-09 07:00:01 | 2015-10-09 14:32:01 | 452.00 | 0 | 0 | 0.0016 |
| `098A72A5-E3E5-4F54-A152-BBDA0DF7B694` | 2015-08-05 04:48:09 | 2015-08-05 10:15:10 | 327.02 | 0 | 0 | 0.0040 |
| `098A72A5-E3E5-4F54-A152-BBDA0DF7B694` | 2015-08-05 10:30:10 | 2015-08-05 14:40:10 | 250.00 | 0 | 0 | 0.0032 |
| `098A72A5-E3E5-4F54-A152-BBDA0DF7B694` | 2015-08-05 17:23:09 | 2015-08-05 19:27:09 | 124.00 | 6 | 0 | 0.0131 |

### Univariate Analysis

<iframe src="assets/fig_uni_duration.html" width="100%" height="520" frameborder="0"></iframe>

**Interpretation**: The distribution of nocturnal sleep duration across 224 sessions is roughly unimodal and centered near 370–400 minutes (~6.2 to 6.7 hours), with a median of 380 minutes. Most sleep episodes fall between 5 and 8 hours. The absence of extreme outliers confirms that filtering out sessions outside the 2-to-14 hour window isolated genuine sleep blocks without preserving sensor dropouts.

<iframe src="assets/fig_uni_activity.html" width="100%" height="520" frameborder="0"></iframe>

**Interpretation**: Pre-bed physical activity is heavily right-skewed. The majority of sleep sessions involve between 0 and 5 active minutes during the 2-hour pre-bed window, reflecting typical sedentary wind-down routines. Only a small subset of nights show sustained physical movement (15 to 45+ minutes), justifying an operational cutoff ($\ge 15$ active minutes) to identify active evenings.

### Bivariate Analysis

<iframe src="assets/fig_bi_box.html" width="100%" height="520" frameborder="0"></iframe>

**Interpretation**: Comparing sleep durations between sessions with pre-bed exercise ($\ge 15$ mins) and sedentary sessions reveals remarkably similar central tendencies: both groups have median sleep durations around 6.5 hours (370–390 minutes) with overlapping interquartile ranges. Pre-bed movement does not appear to truncate sleep opportunity in naturalistic conditions.

<iframe src="assets/fig_bi_scatter.html" width="100%" height="520" frameborder="0"></iframe>

**Interpretation**: Bedtime start hour correlates negatively with nocturnal sleep duration across both active and inactive groups: participants retiring in the early morning hours (2:00 AM to 6:00 AM) achieve shorter total sleep, likely due to social obligations and morning daylight. Pre-bed active nights appear distributed across various hours rather than concentrating solely at late hours.

### Interesting Aggregates: Schedule Interactions
Grouping sleep sessions by pre-bed activity and day type (weekend vs. weekday) highlights social schedule dynamics:

| Activity Group | Day Type | Session Count | Mean Duration (min) | Median Duration (min) | Std Duration (min) | Mean Restlessness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Active ($\ge 15$ mins)** | Weekday (Sun–Thu) | 9 | 405.29 | 442.00 | 100.16 | 0.00 |
| **Active ($\ge 15$ mins)** | Weekend (Fri/Sat) | 8 | 329.28 | 276.89 | 178.77 | 0.00 |
| **Inactive ($< 15$ mins)** | Weekday (Sun–Thu) | 142 | 391.16 | 401.51 | 133.73 | 0.01 |
| **Inactive ($< 15$ mins)** | Weekend (Fri/Sat) | 65 | 347.05 | 361.47 | 133.92 | 0.01 |

**Takeaways**:
- On weekdays, pre-bed exercisers achieved slightly longer average sleep (405.29 mins) and substantially higher median sleep (442.00 mins) than sedentary peers (391.16 mins).
- On weekends, pre-bed active sessions exhibited shorter average duration (329.28 mins) and higher variance, likely driven by social night-life disruptions.
- Mean phone restlessness remained near $0.00$ to $0.01$ across all groups, indicating that evening physical exertion did not induce measurable tossing and turning during sleep.

---

## Assessment of Missingness

### MNAR Analysis
In the ExtraSensory dataset, **`label:SLEEPING`** is missing in approximately 24.4% of recorded intervals. This column is **MNAR (Missing Not at Random)**:
- **Why it is MNAR**: Participants logged activities by selecting interface checkmarks or through passive background sampling. When a participant falls asleep, they are physiologically unconscious and cannot interact with the smartphone to confirm or log their sleeping state. The probability of missingness depends directly on the unobserved state itself (the fact that the individual is asleep).
- **Data to make it MAR**: To explain this missingness with observed variables (making it MAR), we would need auxiliary continuous telemetry independent of user consciousness, such as a bedside microphone recording ambient decibel levels, an automated log tracking screen lock duration without touch events, or a smart charging dock sensor.

### Missingness Dependency: Test 1 (Dependent — MAR)
I evaluated whether screen brightness missingness (`lf_measurements:screen_brightness`) depends on battery charging status (`discrete:battery_state:is_charging`):
- **Null Hypothesis ($H_0$)**: Battery charging status is distributed identically whether screen brightness is missing or present (missingness is independent of charging).
- **Alternative Hypothesis ($H_1$)**: Battery charging status differs between rows where screen brightness is missing versus present.
- **Test Statistic**: Absolute difference in charging proportions ($|\bar{X}_{\text{missing}} - \bar{X}_{\text{present}}|$).
- **Significance Level**: $\alpha = 0.05$.

<iframe src="assets/fig_perm_charging.html" width="100%" height="520" frameborder="0"></iframe>

**Interpretation**: The observed absolute difference in charging proportions was **0.0419** (23.24% charging when missing vs. 27.42% charging when present). Across 500 permutation trials, the empirical $p$-value was **0.0000**. Because $p < 0.05$, we reject the null hypothesis. The missingness of screen brightness strongly depends on whether the device is charging, confirming it is **MAR** with respect to battery status.

### Missingness Dependency: Test 2 (Independent)
I evaluated whether screen brightness missingness depends on weekend status (`is_weekend`):
- **Null Hypothesis ($H_0$)**: The distribution of `is_weekend` is the same whether screen brightness is missing or present.
- **Alternative Hypothesis ($H_1$)**: The distribution of `is_weekend` differs depending on whether screen brightness is missing.
- **Test Statistic**: Absolute difference in weekend proportions ($|\bar{X}_{\text{missing}} - \bar{X}_{\text{present}}|$).
- **Significance Level**: $\alpha = 0.05$.

<iframe src="assets/fig_perm_weekend.html" width="100%" height="520" frameborder="0"></iframe>

**Interpretation**: The observed absolute difference in weekend proportions was **0.0162** (29.95% weekend when missing vs. 28.33% weekend when present). Across 500 permutation trials on a representative 1,000-row sample, the empirical $p$-value was **0.6320**. Because $p \ge 0.05$, we fail to reject the null hypothesis. Screen brightness missingness **does not depend** on weekend status, fulfilling the requirement to demonstrate an independent missingness relationship.

---

## Hypothesis Testing

To evaluate whether naturalistic evening physical exertion alters nocturnal sleep duration, I conducted a two-sample permutation test:

- **Null Hypothesis ($H_0$)**: Sleep duration for sessions preceded by exercise ($\ge 15$ active mins in the 2-hour window) and sessions without pre-bed exercise originate from the same underlying distribution. Any observed difference in means is due to random chance.
- **Alternative Hypothesis ($H_1$)**: Sleep duration for sessions preceded by exercise comes from a different distribution than sessions without pre-bed exercise (two-tailed).
- **Test Statistic**: Difference in group sample means ($\bar{X}_{\text{active}} - \bar{X}_{\text{inactive}}$).
- **Significance Level**: $\alpha = 0.05$.

<iframe src="assets/fig_hyp.html" width="100%" height="520" frameborder="0"></iframe>

**Statistical Decision**:
- **Observed Difference**: Sessions preceded by exercise averaged **369.52 minutes** (~6.16 hours), while inactive sessions averaged **377.31 minutes** (~6.29 hours)—an observed difference of **-7.79 minutes**.
- **Permutation Results**: Shuffling activity indicators across 1,000 trials produced a null distribution centered at 0 with differences ranging between -40 and +40 minutes. The observed difference of -7.79 minutes produced a two-sided empirical $p$-value of **0.8180**.
- **Conclusion**: Because $p = 0.8180 \ge 0.05$, I **fail to reject the null hypothesis**. The data provides insufficient evidence to claim that pre-bed physical activity significantly shortens or lengthens sleep duration in naturalistic settings. The clinical concern that evening workouts cut sleep short is not supported by this dataset.

---

## Framing a Prediction Problem

### Problem Formulation
- **Task**: Binary Classification.
- **Objective**: Predict whether an upcoming nocturnal sleep session will result in **short sleep** (defined as total duration $< 390$ minutes, or 6.5 hours).
- **Target Variable**: `is_short_sleep` ($1$ if `duration_mins < 390`, $0$ otherwise).
- **Practical Application**: This model mimics an intelligent sleep notification engine for wearables. If evening sensor telemetry indicates elevated risk of curtailed sleep, the device can proactively deliver wind-down guidance or suggest adjusted morning alarms.

### Time of Prediction ($t = 0$) & Preventing Data Leakage
The prediction is made at **bedtime** (`sleep_start`):
- All feature transformations strictly use sensor data and behavioral records collected **before or at sleep onset**.
- Variables recorded *during* sleep (e.g., in-sleep accelerometer standard deviation, nocturnal wakeups, or wake-up time) are strictly withheld from the feature matrix to eliminate data leakage.

### Evaluation Metric Choice
<iframe src="assets/fig_target.html" width="100%" height="480" frameborder="0"></iframe>

The target variable is balanced (114 standard sleep sessions at 50.9% vs. 110 short sleep sessions at 49.1%). However, I prioritized **F1-Score** (primary) and **ROC-AUC** (secondary) over raw accuracy:
- **F1-Score**: Balances precision (avoiding false alarms that erode user trust) and recall (reliably flagging nights at risk of sleep deprivation) on the short-sleep class.
- **ROC-AUC**: Measures the classifier's capability to rank high-risk nights above low-risk nights across all probability thresholds.

---

## Baseline Model

### Architecture & Features
The baseline classification pipeline utilizes two features available at sleep onset:
1. `pre_bed_active_mins` (Quantitative continuous): Total active minutes in the 2-hour pre-bed window.
2. `start_hour` (Quantitative discrete / Ordinal): The integer hour (0–23) of sleep onset.

### Pipeline & Performance
The data was split into **80% training (179 sessions)** and **20% testing (45 sessions)** using stratified sampling. The model encapsulates numeric feature scaling via `StandardScaler` and a shallow `DecisionTreeClassifier(max_depth=3, random_state=42)` in a single scikit-learn `Pipeline`:

- **Test Accuracy**: 0.6222
- **Test F1-Score**: 0.6047
- **Test ROC-AUC**: 0.6057

<iframe src="assets/fig_cm_base.html" width="100%" height="480" frameborder="0"></iframe>

### Baseline Limitations
While achieving 62.2% test accuracy, the baseline tree suffers from two architectural limitations:
1. **Midnight Discontinuity**: Treating `start_hour` as a linear variable fails to capture the circularity of time; 11:00 PM (`23`) and 1:00 AM (`1`) are treated as distant extremes despite being separated by only two hours.
2. **Ignoring Calendar Rhythms**: The baseline model cannot distinguish between a late bedtime on Tuesday (constrained by early alarms) and a late bedtime on Friday (with opportunity for recovery sleep).

---

## Final Model

### Feature Engineering Rationale
To address baseline deficiencies, I engineered three domain-specific features:
1. **`day_of_week` (Categorical — One-Hot Encoded)**: Work and academic schedules dictate weekday wake times, whereas weekends permit unconstrained sleep. One-hot encoding the day names allows the model to learn day-specific sleep duration baselines.
2. **`is_late_bedtime` (Binary Indicator)**: Solves the midnight discontinuity by flagging bedtimes between 11:00 PM and 4:00 AM (`start_hour >= 23` or `start_hour <= 4`), grouping delayed sleep phase onsets together.
3. **`high_activity_flag` (Binary Indicator)**: Distinguishes casual evening walking from concentrated physical exercise by thresholding activity at $\ge 30$ minutes.

### Hyperparameter Tuning & Model Selection
I upgraded to a `RandomForestClassifier` to capture non-linear interactions across calendar days and bedtime flags. Prior to model fitting, I selected the following hyperparameters for 5-fold cross-validation via `GridSearchCV`:
- `max_depth` ([3, 6, 9]): Regulates tree capacity to prevent individual trees from memorizing outlier participant schedules.
- `min_samples_split` ([2, 5]): Enforces minimum node sizes, stabilizing splits against small sample variations.
- `n_estimators` ([50, 100, 150]): Determines the number of ensemble trees required for variance reduction to saturate.

The grid search optimized for **`f1`**, yielding best parameters: `max_depth=6`, `min_samples_split=5`, and `n_estimators=150` (CV F1: 0.5736).

### Performance & Model Comparison
Evaluating the tuned Random Forest on the identical held-out test set produced:
- **Test Accuracy**: 0.6000
- **Test F1-Score**: 0.5909
- **Test ROC-AUC**: 0.6779

<iframe src="assets/fig_cm_final.html" width="100%" height="480" frameborder="0"></iframe>

<iframe src="assets/fig_comp.html" width="100%" height="520" frameborder="0"></iframe>

**Comparison Takeaways**:
While raw Accuracy and F1 remained roughly flat (0.62 vs. 0.60 and 0.60 vs. 0.59), **ROC-AUC improved substantially from 0.6057 to 0.6779 (+0.072)**. The Random Forest generates significantly better calibrated, continuous probability estimates for sleep deficit risk than the shallow decision tree, establishing a stronger foundation for mobile health threshold alerts.

---

## Fairness Analysis

### Group Definitions & Evaluation
I conducted a fairness assessment to determine whether the final model performs equitably across different social schedules:
- **Group X (Standard / Weekday)**: Sleep sessions on Sunday through Thursday nights ($n = 33$ in test set).
- **Group Y (Context Shift / Weekend)**: Sleep sessions on Friday and Saturday nights ($n = 12$ in test set).
- **Evaluation Metric**: **Accuracy** (overall correct prediction proportion).
- **Null Hypothesis ($H_0$)**: The model is fair; its prediction accuracy is the same for weekend sessions and weekday sessions ($\text{Acc}_{\text{weekend}} - \text{Acc}_{\text{weekday}} = 0$).
- **Alternative Hypothesis ($H_1$)**: The model is unfair; its prediction accuracy differs between weekend and weekday sessions ($\text{Acc}_{\text{weekend}} - \text{Acc}_{\text{weekday}} \neq 0$).
- **Test Statistic**: Absolute difference in accuracy ($|\text{Accuracy}_{\text{weekend}} - \text{Accuracy}_{\text{weekday}}|$).
- **Significance Level**: $\alpha = 0.05$.

<iframe src="assets/fig_fairness.html" width="100%" height="520" frameborder="0"></iframe>

### Fairness Decision & Takeaways
- **Observed Disparity**: The model achieved an accuracy of **0.6667** (66.7%) on weekdays and **0.4167** (41.7%) on weekends, yielding an observed absolute difference of **0.2500**.
- **Permutation Results**: Shuffling weekend labels across test records over 1,000 trials produced an empirical $p$-value of **0.1660**.
- **Statistical Conclusion**: Because $p = 0.1660 \ge 0.05$, I **fail to reject the null hypothesis**. While an absolute accuracy gap of 25% appears substantial, the weekend test set contains only 12 observations; a shift in only two or three predictions accounts for this difference. There is insufficient statistical evidence to conclude the model is systematically unfair across schedule types.
