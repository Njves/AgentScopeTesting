"""Рисуем графики для ноутбука о жилых домах Красноярска."""

from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


floors = 'Этажность'
stop = 'Ближайшая остановка, м'
address = 'Адрес дома'
coordinates = 'Координаты'
blue, teal, amber, ink, grid = '#2964A3', '#087E77', '#BE5B2B', '#172D45', '#DFE7EF'


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


def district_bar(dashboard, values, title, color, row, col, *, showlegend=True, fmt=',.0f'):
    dashboard.add_trace(go.Bar(
        x=values, y=values.index, orientation='h', name=title,
        marker_color=color, showlegend=showlegend,
        hovertemplate=f'%{{y}}<br>{title}: %{{x:{fmt}}}<extra></extra>',
    ), row, col)
