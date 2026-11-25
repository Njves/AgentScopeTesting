"""Build the house catalog and simulated survey responses from the source workbook."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PLACES = {
    'stop': ('Ближайшая остановка, м', 'Остановка'),
    'school': ('Школа, м', 'Школа'),
    'kindergarten': ('Детский сад, м', 'Детсад'),
    'pharmacy': ('Аптека, м', 'Аптека'),
    'hospital': ('Больница, м', 'Больница'),
    'park': ('Парк, м', 'Парк'),
    'mall': ('ТЦ, м', 'ТЦ'),
}
DISTANCES = [f'{place}_m' for place in PLACES]
LIMITS = [f'want_{place}_m' for place in PLACES]
PRIORITIES = [f'priority_{place}' for place in (*PLACES, 'parking')]
FEATURES = DISTANCES + ['parking'] + LIMITS + PRIORITIES
PARKING = ['До 150 м', '150–400 м', 'Дальше 400 м']
DISTRICTS = [
    'Железнодорожный', 'Кировский', 'Ленинский', 'Октябрьский',
    'Свердловский', 'Советский', 'Центральный',
]
TARGET = 'shortlisted'
SEED = 42
SURVEY_COUNT = 600
HOUSES_PER_SURVEY = 30


def read_catalog(source):
    columns = {
        '№': 'house_id', 'Адрес дома': 'address', 'Район': 'district',
        'Этажность': 'floors', 'Координаты': 'coordinates', 'Парковка': 'parking',
        **{PLACES[place][0]: f'{place}_m' for place in PLACES},
    }
    raw = pd.read_excel(source, sheet_name='Данные', usecols=list(columns)).rename(columns=columns)
    city = raw.loc[raw['district'].isin(DISTRICTS)].copy()
    city['address'] = city['address'].astype('string').str.replace(r'\s+', ' ', regex=True).str.strip()
    house_number = (
        r'\s\d+[а-яёa-z]?(?:[/\-]\d+[а-яёa-z]?)*'
        r'(?:\s+(?:корпус|к|строение|стр|с|лит)\.?\s*(?:\d+[а-яёa-z]?|[а-яёa-z]))?$'
    )
    full_address = city['address'].str.contains(house_number, case=False, na=False)
    full_address &= ~city['address'].str.match(r'^дом\s', case=False, na=False)
    catalog = city.loc[full_address].copy()

    coordinates = catalog['coordinates'].str.split(',', n=1, expand=True)
    catalog['latitude'] = pd.to_numeric(coordinates[0], errors='coerce')
    catalog['longitude'] = pd.to_numeric(coordinates[1], errors='coerce')
    catalog[DISTANCES] = catalog[DISTANCES].apply(pd.to_numeric, errors='coerce')
    catalog['parking'] = catalog['parking'].replace({
        'наземная парковка до 150 м': PARKING[0],
        'парковка 150–400 м': PARKING[1],
        'далее 400 м': PARKING[2],
    })
    floors = pd.to_numeric(catalog['floors'], errors='coerce')
    catalog['floors'] = floors.where(floors.ge(1) & floors.mod(1).eq(0)).astype('Int64')

    valid = (
        catalog['latitude'].between(-90, 90) & catalog['longitude'].between(-180, 180)
        & np.isfinite(catalog[DISTANCES]).all(axis=1) & catalog[DISTANCES].ge(0).all(axis=1)
        & catalog['parking'].isin(PARKING)
    )
    valid_houses = catalog.loc[valid].assign(address_key=lambda frame: frame['address'].str.casefold())
    duplicate = valid_houses.duplicated(['district', 'address_key'])
    rows = ['house_id', 'address', 'district', 'floors', 'latitude', 'longitude', *DISTANCES, 'parking']
    houses = valid_houses.loc[~duplicate, rows].sort_values('house_id').reset_index(drop=True)
    audit = pd.Series({
        'Строк в исходной книге': len(raw),
        'Строк вне семи районов': len(raw) - len(city),
        'Строк без распознаваемого номера дома': len(city) - len(catalog),
        'Некорректных обязательных полей': int((~valid).sum()),
        'Повторов адреса в пределах района': int(duplicate.sum()),
        'Адресов в каталоге': len(houses),
    })
    assert houses['house_id'].is_unique and set(houses['district']) == set(DISTRICTS)
    return houses, audit


def generate_surveys(houses, rng):
    profiles = [
        ('school', 'kindergarten'), ('pharmacy', 'hospital'),
        ('stop', 'mall'), ('park', 'parking'),
    ]
    importance = rng.choice(4, size=(SURVEY_COUNT, len(PRIORITIES)), p=[.22, .28, .30, .20])
    for index, profile in enumerate(rng.integers(len(profiles), size=SURVEY_COUNT)):
        columns = [PRIORITIES.index(f'priority_{place}') for place in profiles[profile]]
        importance[index, columns] = rng.integers(2, 4, size=len(columns))

    surveys = pd.DataFrame({
        'survey_id': np.arange(1, SURVEY_COUNT + 1),
        'districts': ['|'.join(sorted(rng.choice(DISTRICTS, rng.integers(1, 4), replace=False)))
                      for _ in range(SURVEY_COUNT)],
    })
    # The catalog quartiles give three realistic choices at each distance scale.
    for distance, limit in zip(DISTANCES, LIMITS):
        options = houses[distance].quantile([.25, .50, .75]).to_numpy()
        options = np.maximum(50, 50 * np.rint(options / 50)).astype(int)
        surveys[limit] = rng.choice(options, size=SURVEY_COUNT)
    surveys[PRIORITIES] = importance
    return surveys


def generate_answers(houses, surveys, rng):
    parking_utility = dict(zip(PARKING, [.85, .20, -.85]))
    district_indices = houses.groupby('district').indices
    education_columns = [DISTANCES.index(name) for name in ['school_m', 'kindergarten_m']]
    healthcare_columns = [DISTANCES.index(name) for name in ['pharmacy_m', 'hospital_m']]
    blocks = []
    for _, survey in surveys.iterrows():
        pool = np.concatenate([district_indices[name] for name in survey['districts'].split('|')])
        chosen = houses.iloc[rng.choice(pool, size=HOUSES_PER_SURVEY, replace=False)]
        limits = survey[LIMITS].to_numpy(dtype=float)
        weights = survey[PRIORITIES].to_numpy(dtype=float)
        ratio = chosen[DISTANCES].to_numpy(dtype=float) / limits
        utilities = (1 - ratio) / (1 + ratio)
        parking = chosen['parking'].map(parking_utility).to_numpy(dtype=float)
        utility = np.average(np.column_stack([utilities, parking]), weights=weights, axis=1)

        education = (weights[education_columns] >= 2).all() & (ratio[:, education_columns] <= 1).all(axis=1)
        health = (weights[healthcare_columns] >= 2).all() & (ratio[:, healthcare_columns] <= 1).all(axis=1)
        strictness = rng.normal(0, .55)
        logit = -.65 + 4.2 * utility + .65 * education + .35 * health - strictness
        probability = 1 / (1 + np.exp(-logit))
        blocks.append(pd.DataFrame({
            'survey_id': survey['survey_id'],
            'house_id': chosen['house_id'].to_numpy(),
            TARGET: rng.binomial(1, probability),
        }))

    answers = pd.concat(blocks, ignore_index=True).sort_values(['survey_id', 'house_id'])
    assert not answers.duplicated(['survey_id', 'house_id']).any()
    assert answers.groupby('survey_id').size().eq(HOUSES_PER_SURVEY).all()
    return answers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Original XLSX workbook with a «Данные» sheet')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('data'))
    args = parser.parse_args()

    houses, audit = read_catalog(args.source)
    rng = np.random.default_rng(SEED)
    surveys = generate_surveys(houses, rng)
    answers = generate_answers(houses, surveys, rng)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in [('houses', houses), ('surveys', surveys), ('answers', answers)]:
        table.to_csv(args.output / f'{name}.csv', index=False, encoding='utf-8-sig')
    print(audit.to_string())
    print(f'Анкет: {len(surveys)}. Оценок: {len(answers)}. Ответов «да»: {answers[TARGET].mean():.1%}.')


if __name__ == '__main__':
    main()
