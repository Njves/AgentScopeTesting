"""Загрузка данных и графики о жилых домах Красноярска."""

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


floors = 'Этажность'
stop = 'Ближайшая остановка, м'
address = 'Адрес дома'
coordinates = 'Координаты'
target = 'Тип жильца'
district = 'Район'
court = 'Закрытый двор'
air = 'Загрязнение воздуха, % (модель)'
profile_distances = ['Школа, м', 'Детский сад, м', stop, 'Развлечения, м', 'Аптека, м', 'Больница, м']
blue, teal, amber, ink, grid = '#2964A3', '#087E77', '#BE5B2B', '#172D45', '#DFE7EF'
class_colors = {
    'Семейная': blue, 'Для одного человека': teal,
    'Для пожилых людей': '#7764A5', 'Для автомобилиста': amber,
}


def load_data():
    path = Path(__file__).with_name('датасет.xlsx')
    raw = pd.read_excel(path, sheet_name='Данные', keep_default_na=False)
    data = raw.replace(r'^\s*(?:информация отсутствует)?\s*$', np.nan, regex=True)
    numeric = [col for col in raw if col.endswith(', м')] + [floors, air]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors='coerce')
    data[court] = np.nan  # Сведения об ограждениях отсутствуют по методике источника.
    if data[target].isna().any() or not data[target].isin(class_colors).all():
        raise ValueError('Столбец «Тип жильца» содержит пропуски или неизвестные классы.')
    return raw, data


def read_coordinates(data):
    geo = data[coordinates].str.split(',', n=1, expand=True).apply(pd.to_numeric, errors='coerce')
    geo.columns = ['Широта', 'Долгота']
    valid = geo['Широта'].between(-90, 90) & geo['Долгота'].between(-180, 180)
    return geo.loc[valid]


def style(fig, height):
    fig.update_layout(
        template='plotly_white', height=height, paper_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=13, color=ink),
        margin=dict(l=65, r=55, t=65, b=55), separators=', ',
    )
    return fig


def panels(columns, draw, short, *, ncols=4, height=2.35, **draw_options):
    nrows = (len(columns) + ncols - 1) // ncols
    _, axes = plt.subplots(
        nrows, ncols, figsize=(13, nrows * height), layout='constrained', squeeze=False,
    )
    for ax, col in zip(axes.flat, columns):
        draw(ax, col, **draw_options)
        ax.set_title(fill(short[col], 28, break_long_words=False), loc='left')
    for ax in axes.flat[len(columns):]:
        ax.set_visible(False)
    plt.show()


def histogram(ax, col, *, data):
    values = data[col].dropna()
    if col == stop and not values.empty:
        step = 500
        bins = np.arange((int(values.min()) // step) * step,
                         (int(values.max()) // step + 2) * step, step)
        counts, edges, bars = ax.hist(values, bins=bins, color=blue, rwidth=.92)
        bars[0].set_facecolor(teal)
        ax.set_ylim(top=counts.max() * 1.25)
        ax.annotate(
            f'{edges[0]:.0f}–{edges[1] - 1:.0f} м\nМинимум — {values.min():.0f} м',
            xy=(edges[0] + step / 2, counts[0]), xycoords='data',
            xytext=(.31, .95), textcoords='axes fraction', va='top', fontsize=10,
            arrowprops=dict(arrowstyle='->', color=ink),
        )
    else:
        # Для этажности нужен отдельный столбик на каждый этаж.
        bins = np.arange(values.min() - .5, values.max() + 1.5) if col == floors else 24
        ax.hist(values, bins=bins, color=blue)
    ax.grid(axis='y', color=grid)
    if col.endswith(', м'):
        ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('Записей')


labels = {
    'информация отсутствует': 'нет сведений',
    'дерево/частный дом (по типу здания)': 'дерево (по типу)',
    'кирпич/газобетон (малоэтажный, оценка)': 'кирпич/газобетон',
    'панель/монолит-жб (многоэтажный, по этажности — оценка)': 'панель/монолит',
    'кирпич/панель (среднеэтажный, оценка)': 'кирпич/панель',
    'далее 400 м (нет парковки в данных)': 'далее 400 м',
    'наземная парковка рядом (до 150 м)': 'до 150 м',
    'парковка в радиусе 400 м': '150–400 м',
}


def category(ax, col, *, geo, raw, data):
    if col == coordinates:
        ax.scatter(geo['Долгота'], geo['Широта'], s=2, alpha=.25, color=teal)
        ax.set(xlabel='Долгота, °', ylabel='Широта, °')
        return
    if col == '№':
        ax.hist(raw[col], bins=12, color=blue)
        ax.set(xlabel='Номер записи', ylabel='Записей')
        return
    if col == address:
        counts = data[col].value_counts()
        frequencies = counts.clip(upper=3).value_counts()
        values = pd.Series({
            '1 запись': frequencies.get(1, 0),
            '2 записи': frequencies.get(2, 0),
            '3+ записи': frequencies.get(3, 0),
        })
        ax.set_xlabel('Различных адресов')
    else:
        values = raw[col].replace(labels).value_counts().rename(
            index={'нет (двор без ограждения в данных)': 'нет сведений о дворе'}
        )
        if len(values) > 5:
            rest = values.iloc[4:].sum()
            values = values.iloc[:4].copy()
            values.loc['прочие'] = rest
        ax.set_xlabel('Записей')
    values = values.iloc[::-1]
    bars = ax.barh([fill(str(label), 17) for label in values.index], values, color=blue)
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.margins(x=.18)
    ax.tick_params(axis='y', labelsize=8, length=0)
    ax.grid(axis='x', color=grid)


def box(ax, col, *, data):
    ax.boxplot(
        data[col].dropna(), orientation='horizontal', widths=.48, patch_artist=True,
        boxprops=dict(facecolor='#CFDFEF', edgecolor=blue),
        medianprops=dict(color=ink, linewidth=2),
        flierprops=dict(marker='o', markersize=2.3, markeredgecolor=amber, alpha=.4),
    )
    ax.set_yticks([])
    ax.grid(axis='x', color=grid)


def profile_dashboard(data, geo):
    counts = data[target].value_counts().reindex(class_colors)
    medians = data.groupby(target)[profile_distances].median().reindex(counts.index)
    covered = data[district].value_counts().loc[lambda n: n >= 30].index
    district_counts = pd.crosstab(data[district], data[target]).reindex(
        index=covered, columns=counts.index, fill_value=0,
    )
    shares = district_counts.div(district_counts.sum(axis=1), axis=0)
    points = data.join(geo, how='inner')

    fig = make_subplots(
        rows=3, cols=2, row_heights=[.33, .39, .28],
        specs=[[{}, {}], [{}, {}], [{'type': 'table', 'colspan': 2}, None]],
        horizontal_spacing=.18, vertical_spacing=.115,
        subplot_titles=['Размер классов', 'Доли классов по районам',
                        'Дома по координатам', 'Расстояние до остановки',
                        'Типичные расстояния по классам, м'],
    )
    fig.add_trace(go.Bar(
        x=counts, y=counts.index, orientation='h', marker_color=list(class_colors.values()),
        showlegend=False, hovertemplate='%{y}<br>%{x:,} записей<extra></extra>',
    ), 1, 1)
    for label, color in class_colors.items():
        fig.add_trace(go.Bar(
            x=shares[label], y=shares.index, orientation='h', name=label,
            marker_color=color, customdata=district_counts[label], legendgroup=label,
            hovertemplate='%{y}<br>%{x:.1%} (%{customdata:,} записей)',
        ), 1, 2)
        group = points.loc[points[target].eq(label)]
        fig.add_trace(go.Scatter(
            x=group['Долгота'], y=group['Широта'], text=group[address],
            mode='markers', marker=dict(size=4, opacity=.65, color=color),
            name=label, legendgroup=label, showlegend=False,
            hovertemplate='%{text}<extra>%{fullData.name}</extra>',
        ), 2, 1)
        fig.add_trace(go.Box(
            x=data.loc[data[target].eq(label), stop], y0=label, orientation='h',
            name=label, marker_color=color, legendgroup=label, showlegend=False, boxpoints=False,
        ), 2, 2)

    fig.add_vline(x=4200, line_dash='dot', line_color=amber,
                  annotation_text='Порог 4 200 м', row=2, col=2)
    table = medians.rename(columns={stop: 'Остановка, м'}).map(lambda value: f'{value:g}').reset_index()
    fig.add_trace(go.Table(
        columnwidth=[1.65, 1, 1, 1, 1, 1, 1],
        header=dict(values=list(table), fill_color=ink, font=dict(color='white'), height=29, align='left'),
        cells=dict(values=table.T.to_numpy().tolist(), fill_color=['#EFF5FB'], height=26, align='left'),
    ), 3, 1)

    fig.update_xaxes(title='Записей', row=1, col=1)
    fig.update_xaxes(tickformat='.0%', range=[0, 1], row=1, col=2)
    fig.update_xaxes(title='Долгота, °', row=2, col=1)
    fig.update_yaxes(title='Широта, °', row=2, col=1)
    fig.update_xaxes(title='Расстояние по прямой, м', row=2, col=2)
    for row, col in [(1, 1), (1, 2), (2, 2)]:
        fig.update_yaxes(autorange='reversed', row=row, col=col)
    return style(fig, 835).update_layout(
        title=dict(text='Тип жильца и доступность инфраструктуры в Красноярске', y=.975, yanchor='top'),
        barmode='stack', margin=dict(l=145, r=40, t=132, b=42),
        legend=dict(orientation='h', y=1.055, yanchor='bottom'),
    )
