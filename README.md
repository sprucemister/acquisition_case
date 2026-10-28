# Purpose

Investigate 3 attribution channel for subscription sales. Make baseline attribution numbers. Then, make a more advanced model for one of the channels.

## How To Run

### Package Management

Utilizes [uv](https://docs.astral.sh/uv/) for package management

```powershell
uv sync
uv run run-analysis --data-dir path\to\data # real data
```

Recomended to put raw data files in a folder in the root directory called 'data'

### AI Usage

AI was used to modularize the code

You can alternatively see and run the raw code in the file `Main - Rough.py` using the same uv environment

## Approach - Baseline

### Digital Interactions

- Use direct links to sales from `cookie_id` 
- Remove digital interactions older than 7 days before sale
- Don't credit Paid Search, Organic Search, and Direct are 

### DM Sendouts

- Use direct links to sales from `address_id` 
- Use the holdout group's conversion rate to calculate a incremental lift

### TV Spot

- Define `tv_spot_active` as the calender day any spot aired and the next calender day
- Take the average difference in sales between when TV Spots are active and not
- Multiply that by the number of days active

## Approach - Deep Dive into Digital Interactions

### Key Insights

- Seasons of September-October, May-August, and November-April have similar patterns of both sales and advertising across different channels. Examples include:

   - September and October have very high sales and TV Spot impressions as well as an increase in Digital Interactions (in particular - Paid Social)

   - May-August have no TV Spots, the also represent a dip in sales and digital interactions (in particular - Paid Social)

- After DM Sendouts, sales directly attributable to them increase rapidly to a peak around day 5

   - Sales then fall rapidly until around day 8 where they then fall off more gradually until the effect of the DM Sendout is minimal

### Method

- Use day 27 as a proxy for minimal effect of the DM Sendout

- For each of the 3 seasons, take that as a threshold to re-distribute sales

- For days 1-27, take sales over that threshold as attributable only to DM Sendout

   - Sales below the threshold are attributable to both DM Sendout and Digital Interactions just as in the baseline attriubtion method

## Layout

```
src/acquisition/
  config.py       paths and assumptions
  utils.py        helpers (dates, saving, calendar)
  load_clean.py   read + clean
  attribution.py  digital mix, baseline, cases
  tv.py           TV analyses
  variants.py     holdout vs. Variant A/B
  explore.py      exploratory charts, hazard, digital mix windows
  model.py        'Both' reallocation
  synthetic.py    demo data
  run_all.py      entry point
```