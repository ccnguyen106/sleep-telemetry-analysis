# Sleep Telemetry Analysis: Can Pre-Bed Workouts Coexist with Restorative Sleep?

**Author**: Christopher Nguyen  
**Institution**: University of California, San Diego  
**Course**: DSC 80: Theoretical Foundations of Data Science  
**Website Link**: [https://ccnguyen106.github.io/sleep-telemetry-analysis/](https://ccnguyen106.github.io/sleep-telemetry-analysis/)

---

## Quick Navigation
- [Introduction](#introduction)
- [Data Cleaning and Exploratory Data Analysis](#data-cleaning-and-exploratory-data-analysis)
- [Assessment of Missingness](#assessment-of-missingness)
- [Hypothesis Testing](#hypothesis-testing)
- [Framing a Prediction Problem](#framing-a-prediction-problem)
- [Baseline Model](#baseline-model)
- [Final Model](#final-model)
- [Fairness Analysis](#fairness-analysis)

---

## Introduction

We've all heard the standard sleep advice: don't work out right before bed because it spikes your heart rate, raises your core temperature, and keeps you awake. The logic makes sense on paper, but in reality, life doesn't always fit an early-evening gym schedule. Between classes, work, and commuting, late night is often the only time people have to walk their dog, go for a run, or hit the gym. On top of that, newer wearable research suggests light-to-moderate evening movement might actually relieve stress and make it easier to wind down and fall asleep.

Using the UCSD ExtraSensory Dataset, I wanted to see what actually happens to sleep after an active evening in the real world:

> **Primary Research Question**:  
> *Does physical activity in the two hours leading up to bedtime affect nocturnal sleep duration and sleep restlessness?*

### The ExtraSensory Dataset
The raw ExtraSensory dataset contains minute-by-minute sensor readings and self-reported activity logs across 60 unique participants living their everyday lives, totaling **377,346 rows across 279 columns**.

Most of those 279 columns are raw audio features, location coordinates, or frequency transformations that aren't relevant to bedtime habits and would slow down processing. To keep the analysis focused and computationally manageable, I filtered the data down to 14 core columns capturing physical movement, sleep state, phone motion, screen use, and charging status:

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

### Cleaning & Session Extraction Pipeline
The raw ExtraSensory data comes in 1-minute time slices. But my research question isn't about individual minutes—it's about full sleep sessions and what happens leading up to them. Here is how I cleaned and transformed the data:

1. **Datetime Conversion**: Converted the raw integer timestamp into a pandas `datetime64` object (`date_time`) so I could sort chronologically, extract hours, and perform rolling time lookups.
2. **Handling Missing Activity Labels**: In the ExtraSensory collection app, users only checked off activities they were actively doing. An unselected box showed up as `NaN` in the CSV, but that doesn't mean the data was lost—it means the user wasn't walking, running, or exercising at that minute. I filled `NaN` in those three columns with `0.0`.
3. **Compound Activity Indicator**: Created an `is_active` column that equals 1 if a user was walking, running, or exercising, and 0 otherwise.
4. **Isolating Sleep Sessions**: Grouped consecutive minutes where `label:SLEEPING == 1.0` into discrete sleep blocks using `.cumsum()` on the start of each sleep streak.
5. **Session Filtering**: Filtered out any sleep block shorter than 2 hours (120 minutes) or longer than 14 hours (840 minutes). This drops quick daytime naps, brief logging glitches, and multi-day sensor cutoffs, leaving 224 clean nocturnal sleep sessions.
6. **2-Hour Pre-Bed Lookback Window**: For each valid sleep session, I looked back at the 120 minutes prior to `sleep_start`. I summed up the active minutes (`pre_bed_active_mins`), classified anyone with 15+ active minutes as having `exercised_before_bed = 1`, and pulled out the bedtime hour and day of the week.

### Cleaned Dataset Preview
Below are the first 5 rows of the cleaned sleep sessions DataFrame:

| uuid | sleep_start | sleep_end | duration_mins | pre_bed_active_mins | exercised_before_bed | restlessness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `00EABED2-271D-49D8-B599-1D4A09240601` | 2015-10-08 18:02:33 | 2015-10-08 21:50:46 | 228.22 | 0 | 0 | 0.0605 |
| `00EABED2-271D-49D8-B599-1D4A09240601` | 2015-10-09 07:00:01 | 2015-10-09 14:32:01 | 452.00 | 0 | 0 | 0.0016 |
| `098A72A5-E3E5-4F54-A152-BBDA0DF7B694` | 2015-08-05 04:48:09 | 2015-08-05 10:15:10 | 327.02 | 0 | 0 | 0.0040 |
| `098A72A5-E3E5-4F54-A152-BBDA0DF7B694` | 2015-08-05 10:30:10 | 2015-08-05 14:40:10 | 250.00 | 0 | 0 | 0.0032 |
| `098A72A5-E3E5-4F54-A152-BBDA0DF7B694` | 2015-08-05 17:23:09 | 2015-08-05 19:27:09 | 124.00 | 6 | 0 | 0.0131 |

### Univariate Analysis

<iframe src="assets/fig_uni_duration.html" width="100%" height="520" frameborder="0" style="border:none;"></iframe>

**Interpretation**: The distribution of nocturnal sleep duration across the 224 identified sessions is roughly unimodal and centered around 370–400 minutes (about 6 to 6.5 hours). The box plot shows a median of roughly 380 minutes, with the majority of nights falling between 5 and 8 hours. There are a handful of shorter sessions near the 2-hour cutoff and longer sessions near 11–12 hours, but no extreme outliers, confirming that the 2-to-14 hour filtering successfully isolated real overnight sleep episodes without leaving behind sensor dropouts.

<iframe src="assets/fig_uni_activity.html" width="100%" height="520" frameborder="0" style="border:none;"></iframe>

**Interpretation**: This distribution is heavily right-skewed. The vast majority of sleep sessions show between 0 and 5 minutes of active movement in the two hours before bed, which makes sense—most people wind down and remain sedentary (sitting, reading, lying in bed) before falling asleep. Only a small subset of nights show sustained movement (15 to 45+ minutes of walking or exercise). This confirms that pre-bed exercise is relatively rare in naturalistic living and motivates using a threshold (15 minutes) to distinguish active evenings from standard sedentary ones.

### Bivariate Analysis

<iframe src="assets/fig_bi_box.html" width="100%" height="520" frameborder="0" style="border:none;"></iframe>

**Interpretation**: The box plot compares sleep duration between sessions with pre-bed exercise (15+ active minutes) and sessions without it (< 15 active minutes). Interestingly, the distributions look very similar: both groups have median sleep durations right around 6.5 hours (370–390 minutes), and their interquartile ranges heavily overlap. Pre-bed exercisers do show a slightly wider spread on the lower end, but at a glance, evening physical activity does not seem to drastically cut sleep short. I'll test whether this small difference is statistically meaningful in Step 4.

<iframe src="assets/fig_bi_scatter.html" width="100%" height="520" frameborder="0" style="border:none;"></iframe>

**Interpretation**: This scatter plot maps the hour the participant fell asleep (0–23 on the x-axis) against total sleep minutes, broken down by pre-bed activity. We can see two clear clusters: people falling asleep late at night (22:00 to 02:00) and people sleeping during irregular hours. Across both groups, going to bed later in the early morning hours (e.g., 2 AM to 5 AM) correlates with shorter overall sleep, likely because morning alarms or sunlight force people awake. Pre-bed active sessions appear scattered across various bedtimes rather than clustered solely at late hours.

### Interesting Aggregates: Schedule Interactions
To see if weekend schedules mask or change the effect of pre-bed activity, I grouped the sessions by both Activity Group and Day Type (Weekend vs. Weekday):

| Activity Group | Day Type | Session Count | Mean Duration (min) | Median Duration (min) | Std Duration (min) | Mean Restlessness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Active (≥ 15 mins)** | Weekday (Sun–Thu) | 9 | **405.29** | **442.00** | 100.16 | 0.00 |
| **Active (≥ 15 mins)** | Weekend (Fri/Sat) | 8 | **329.28** | **276.89** | 178.77 | 0.00 |
| **Inactive (< 15 mins)** | Weekday (Sun–Thu) | 142 | **391.16** | **401.51** | 133.73 | 0.01 |
| **Inactive (< 15 mins)** | Weekend (Fri/Sat) | 65 | **347.05** | **361.47** | 133.92 | 0.01 |

**Takeaways from the Summary Table**:
- On weekdays, pre-bed exercisers actually slept slightly longer on average (405.3 minutes, or ~6.75 hours) compared to sedentary weekdays (391.2 minutes), with higher median duration (442.0 minutes).
- On weekends, pre-bed exercisers averaged less sleep (329.3 minutes) than sedentary peers (347.1 minutes), though the active weekend sample size is relatively small (n = 8).
- Restlessness (mean accelerometer standard deviation) stayed virtually identical at ~0.00 to 0.01 across all four subgroups, suggesting that evening workouts didn't cause participants to toss and turn significantly more once they actually fell asleep.

---

## Assessment of Missingness

### MNAR Analysis
In the ExtraSensory dataset, **`label:SLEEPING`** is missing in approximately 24.4% of recorded intervals. I believe this column is **MNAR (Missing Not at Random)**:
- **Why it is MNAR**: Participants logged their activities either by actively checking off labels on the phone or through passive sampling. When a participant falls asleep, they are physiologically unconscious and cannot actively interact with the phone to confirm or log that they are sleeping. The probability that the `label:SLEEPING` state is omitted or unverified depends directly on the unobserved state itself (the fact that the participant is asleep).
- **What data would make it MAR**: To explain this missingness using observed data (making it MAR), we would need external sensor streams that don't depend on user consciousness, for example, continuous bedside microphone audio (detecting quiet breathing or ambient room silence), an external bedside charging-dock sensor, or an automated log tracking whether the phone screen remained completely locked without touch events for an extended period.

### Missingness Dependency: Test 1 (Dependent — MAR)
I wanted to see whether screen brightness readings (`lf_measurements:screen_brightness`) are missing at random with respect to whether the phone is plugged in (`discrete:battery_state:is_charging`). People often plug their phone in on a nightstand overnight with the screen turned off, which might lead to missing brightness readings.
- **Null Hypothesis (H₀)**: The distribution of battery charging status is the same whether screen brightness is missing or present. (Missingness is independent of charging).
- **Alternative Hypothesis (H₁)**: The distribution of battery charging status differs depending on whether screen brightness is missing. (Missingness depends on charging).
- **Test Statistic**: Absolute difference in charging proportions (|Charging Rate (Missing) - Charging Rate (Present)|).
- **Significance Level**: α = 0.05.

<iframe src="assets/fig_perm_charging.html" width="100%" height="520" frameborder="0" style="border:none;"></iframe>

**Interpretation**: The observed difference in charging rates between rows with missing vs. present screen brightness was **0.0419** (23.24% charging when missing vs. 27.42% charging when present). Across 500 permutations, the simulated differences never came close to 0.0419, resulting in an empirical p-value of **0.0000**. Because p < 0.05, we reject the null hypothesis. The missingness of screen brightness is definitely dependent on whether the phone is charging, meaning `screen_brightness` is **MAR** with respect to battery state.

### Missingness Dependency: Test 2 (Independent)
I next tested whether the missingness of screen brightness (`lf_measurements:screen_brightness`) depends on whether the measurement occurred on a weekend (`is_weekend`). ExtraSensory's background sensor service runs periodic queries based on an internal system clock rather than calendar day, so missingness in screen brightness should occur at similar rates across weekdays and weekends.
- **Null Hypothesis (H₀)**: The distribution of `is_weekend` is the same whether screen brightness is missing or present. Screen brightness missingness is independent of weekend status.
- **Alternative Hypothesis (H₁)**: The distribution of `is_weekend` differs when screen brightness is missing versus present.
- **Test Statistic**: Absolute difference in weekend proportions (|Weekend Rate (Missing) - Weekend Rate (Present)|).
- **Significance Level**: α = 0.05.

<iframe src="assets/fig_perm_weekend.html" width="100%" height="520" frameborder="0" style="border:none;"></iframe>

**Interpretation**: The observed absolute difference in weekend proportions between rows with missing versus present screen brightness was small (**0.0162**; 29.95% weekend when missing vs. 28.33% weekend when present). The permutation test produced an empirical p-value of **0.6320**, well above our significance threshold. Because p ≥ 0.05, we fail to reject the null hypothesis. We conclude that `screen_brightness` missingness **does not depend** on whether it is a weekend, satisfying the requirement to find an independent column.

---

## Hypothesis Testing

Now I return to the primary research question: Does physical activity before bed affect nocturnal sleep duration?

I ran a two-sample permutation test comparing sleep durations between nights with pre-bed exercise (15+ active minutes) and nights without it:

- **Null Hypothesis (H₀)**: Sleep duration for sessions preceded by exercise and sessions without pre-bed exercise come from the same distribution. Any difference in sample means is purely due to random chance.
- **Alternative Hypothesis (H₁)**: Sleep duration for sessions preceded by exercise comes from a different distribution than sessions without pre-bed exercise (two-tailed).
- **Test Statistic**: Difference in group sample means (Mean Duration Active - Mean Duration Inactive).
- **Significance Level**: α = 0.05.

<iframe src="assets/fig_hyp.html" width="100%" height="520" frameborder="0" style="border:none;"></iframe>

### Interpretation & Statistical Decision
- **Observed Difference**: Sessions with pre-bed exercise averaged **369.52 minutes** (~6.16 hours), while inactive sessions averaged **377.31 minutes** (~6.29 hours)—a difference of **-7.79 minutes**.
- **Permutation Results**: After shuffling group labels 1,000 times, the resulting null distribution centered at 0 with differences regularly ranging between -40 and +40 minutes. The observed -7.79 minute difference fell right near the middle of the distribution, giving a two-sided empirical p-value of **0.8180**.
- **Conclusion**: Because p = 0.8180 ≥ 0.05, I **fail to reject the null hypothesis**. There is no statistically significant evidence that getting 15+ minutes of physical activity in the two hours before bed shortens or lengthens sleep duration in this dataset. The conventional fear that evening workouts drastically cut sleep short is not supported by this data.

---

## Framing a Prediction Problem

### Problem Formulation
- **Task**: Binary Classification.
- **Prediction Objective**: Predict whether an upcoming sleep session will result in short / insufficient sleep (defined as total sleep duration < 390 minutes, or 6.5 hours).
- **Target Variable**: `is_short_sleep` (1 if `duration_mins < 390`, 0 otherwise).
- **Practical Motivation**: Wearables like Apple Watch, Fitbit, or Oura want to give users actionable pre-bedtime advice. If a phone detects an elevated risk of short sleep as the user goes to bed, it can prompt a wind-down reminder or suggest setting an adjusted morning alarm.

### Prediction Timing (t = 0) & Preventing Data Leakage
The prediction happens at bedtime (`sleep_start`). To ensure the model is realistic and avoid data leakage:
- All features must come strictly from sensor readings and behavioral data collected before or at the moment of bedtime.
- No features measured during sleep (such as nocturnal accelerometer movement, restless toss-and-turns, or the wake-up time) can be included in the model, because none of those exist yet when the user falls asleep.

### Metric Selection: F1-Score & ROC-AUC
<iframe src="assets/fig_target.html" width="100%" height="480" frameborder="0" style="border:none;"></iframe>

Even though my two classes ended up roughly balanced (~51% standard sleep vs. ~49% short sleep), I chose **F1-Score** as the primary metric and **ROC-AUC** as the secondary metric:
- **F1-Score**: Balances precision (not constantly crying wolf about bad sleep) and recall (actually catching nights when sleep is cut short).
- **ROC-AUC**: Evaluates how well the classifier ranks risk levels across all possible thresholds, without being tied to a single arbitrary 0.5 cutoff.

---

## Baseline Model

### Feature Selection & Setup
For the baseline model, I used two simple features available right at bedtime:
1. `pre_bed_active_mins` (Quantitative continuous): Total active minutes in the 2-hour pre-bed window.
2. `start_hour` (Quantitative discrete / Ordinal): The hour of the day (0–23) the user went to sleep.

I split the data into **80% training (179 sessions)** and **20% testing (45 sessions)**, stratifying by the target label to keep class proportions identical. I used a scikit-learn `Pipeline` with `StandardScaler` on the numeric features and a shallow `DecisionTreeClassifier(max_depth=3, random_state=42)` as the classifier:

- **Test Accuracy**: 0.6222
- **Test F1-Score**: 0.6047
- **Test ROC-AUC**: 0.6057

<iframe src="assets/fig_cm_base.html" width="100%" height="480" frameborder="0" style="border:none;"></iframe>

### Interpretation & Why It's Limited
Out of 45 test sessions, the baseline decision tree correctly identified 13 short sleep nights (True Positives) and 15 standard sleep nights (True Negatives), but missed 9 short sleep sessions (False Negatives) and had 8 false alarms (False Positives).

The baseline is held back by two main issues:
1. **Midnight Discontinuity**: Using only raw `start_hour` misses the midnight wrap-around: 11:00 PM (`start_hour = 23`) and 1:00 AM (`start_hour = 1`) are treated as opposite ends of a linear scale even though they are only two hours apart.
2. **Missing Calendar Context**: The model has no idea what day of the week it is, meaning it treats a late bedtime on Tuesday night (where a morning alarm is looming) the same as a late bedtime on a Friday night (where the user can sleep in).

---

## Final Model

### Feature Engineering Rationale
To fix the shortcomings of the baseline, I engineered three new features before feeding the data into an ensemble model:
1. **`day_of_week` (Categorical — One-Hot Encoded)**: People sleep very differently on weekends versus weekdays due to work and school schedules. One-hot encoding the day of the week lets the model adjust its baseline expectation for each specific night.
2. **`is_late_bedtime` (Binary Indicator)**: Flags whether the participant went to sleep between 11:00 PM and 4:00 AM (`start_hour >= 23` or `start_hour <= 4`). This solves the midnight boundary problem by explicitly grouping late-night sleep onsets together.
3. **`high_activity_flag` (Binary Indicator)**: Flags whether the user logged 30+ active minutes in the pre-bed window. The linear minute count in the baseline couldn't easily separate 5–10 minutes of casual walking from a concentrated 30+ minute workout session.

### Model Choice & Hyperparameter Tuning
I upgraded to a `RandomForestClassifier` to capture non-linear interactions between day of the week and bedtime timing. Before running `GridSearchCV` with 5-fold cross-validation on the training set, I selected the following hyperparameters to tune:
- `max_depth` ([3, 6, 9]): Tree depth directly regulates model capacity. A shallow depth risks underfitting, while an unrestricted depth allows individual trees to memorize small idiosyncrasies in participant behavior. Tuning this ensures the ensemble learns generalizable bedtime boundaries.
- `min_samples_split` ([2, 5]): Controls the minimum number of samples required to split an internal node. Restricting splits prevents the forest from forming leaves based on one or two outlier sleep nights.
- `n_estimators` ([50, 100, 150]): Increasing the number of trees stabilizes the variance reduction of the bagging ensemble, but reaches diminishing returns. Tuning this identifies where validation performance saturates without unnecessary computation.

The grid search optimized specifically for **`f1`**, selecting: `max_depth=6`, `min_samples_split=5`, and `n_estimators=150` (Best 5-Fold CV F1: **0.5736**).

### Performance & Model Comparison

| Performance Metric | Baseline Model (Decision Tree) | Final Model (Random Forest) | Net Change |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | Single Tree (`max_depth=3`) | Bagged Ensemble (`150` estimators) | Non-linear ensemble |
| **Features Used** | `pre_bed_active_mins`, `start_hour` | Baseline + `day_of_week`, `is_late_bedtime`, `high_activity_flag` | +3 Engineered features |
| **Test Accuracy** | 0.6222 | 0.6000 | -0.0222 |
| **Test F1-Score** | 0.6047 | 0.5909 | -0.0138 |
| **Test ROC-AUC** | **0.6057** | **0.6779** | **+0.0722** |

<iframe src="assets/fig_cm_final.html" width="100%" height="480" frameborder="0" style="border:none;"></iframe>

<iframe src="assets/fig_comp.html" width="100%" height="520" frameborder="0" style="border:none;"></iframe>

**Model Comparison Takeaways**:
While raw Accuracy and F1 remained roughly flat (0.62 vs. 0.60 and 0.60 vs. 0.59), **ROC-AUC improved substantially from 0.6057 to 0.6779 (+0.072)**. 

This jump in ROC-AUC is crucial: it shows that the Random Forest produces much better calibrated, smoother probability scores for sleep deficit risk than the rigid depth-3 decision tree. A higher AUC means the model is meaningfully better at ranking higher-risk nights above lower-risk nights, which is what matters when setting probability thresholds for mobile health notifications.

---

## Fairness Analysis

### Question & Group Definitions
Does my final model perform equally well for people on **weekends** versus **weekdays**?

- **Group X (Weekday Nights)**: Sleep sessions starting Sunday through Thursday night (n = 33 in test set).
- **Group Y (Weekend Nights)**: Sleep sessions starting Friday or Saturday night (n = 12 in test set).
- **Metric**: Accuracy (overall correct prediction rate in each group).
- **Null Hypothesis (H₀)**: The model is fair; its prediction accuracy is the same for weekend and weekday sessions. Any observed difference is due to random test set variation (Accuracy Weekend - Accuracy Weekday = 0).
- **Alternative Hypothesis (H₁)**: The model is unfair; its prediction accuracy differs between weekend and weekday sessions (Accuracy Weekend - Accuracy Weekday ≠ 0).
- **Test Statistic**: Absolute difference in accuracy (|Accuracy Weekend - Accuracy Weekday|).
- **Significance Level**: α = 0.05.

<iframe src="assets/fig_fairness.html" width="100%" height="520" frameborder="0" style="border:none;"></iframe>

### Interpretation of Fairness Test & Conclusion
- **Observed Disparity**: The model achieved an accuracy of **0.6667** (66.7%) on weekdays, but only **0.4167** (41.7%) on weekends, giving an observed absolute difference of **0.2500**.
- **Permutation Results**: I shuffled the weekend/weekday labels across the test set 1,000 times to simulate what accuracy differences we'd see by pure chance. The resulting empirical p-value was **0.1660**.
- **Conclusion**: Because p = 0.1660 ≥ 0.05, I **fail to reject the null hypothesis**. Even though a 25% difference looks large on paper, the weekend test set only contained 12 sessions, meaning a swing of just 2 or 3 predictions easily accounts for this gap. There is insufficient statistical evidence to conclude the model is systematically unfair across days of the week, though collecting more weekend data would be necessary to verify this with higher power.
