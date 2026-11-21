"""Тесты расчётных функций. Запуск: pytest (из корня репозитория)."""
import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from geo_utils import (
    count_by_district,
    density_per_area,
    nearest_distance,
    share_within,
    to_metric,
    within_distance,
)

UTM_EKB = "EPSG:32641"  # зона UTM 41N, Екатеринбург


def layer(coords, crs=UTM_EKB):
    return gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in coords], crs=crs)


def test_to_metric_converts_degrees_to_utm():
    ekb = layer([(60.6, 56.84)], crs="EPSG:4326")
    assert to_metric(ekb).crs.to_epsg() == 32641


def test_to_metric_requires_crs():
    with pytest.raises(ValueError):
        to_metric(gpd.GeoDataFrame(geometry=[Point(0, 0)]))


def test_nearest_distance_in_meters():
    sources = layer([(0, 0)])
    targets = layer([(300, 400)])  # 3-4-5: ровно 500 м
    assert nearest_distance(targets, sources).iloc[0] == pytest.approx(500)


def test_share_within_counts_border_as_inside():
    sources = layer([(0, 0)])
    targets = layer([(100, 0), (500, 0), (600, 0)])
    assert share_within(targets, sources, 500) == pytest.approx(2 / 3)


def test_share_within_rejects_degrees():
    sources = layer([(60.6, 56.84)], crs="EPSG:4326")
    targets = layer([(60.61, 56.84)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="метрической"):
        share_within(targets, sources, 500)


def test_share_within_empty_targets():
    assert share_within(layer([]), layer([(0, 0)]), 500) == 0.0



def test_count_by_district_includes_empty_districts():
    districts = gpd.GeoDataFrame(
        {"name": ["Первый", "Второй", "Третий"]},
        geometry=[
            box(0, 0, 10, 10),
            box(10, 0, 20, 10),
            box(20, 0, 30, 10)
        ],
        crs=UTM_EKB
    )

    points = layer([(1, 1), (11, 1)])

    result = count_by_district(points, districts)

    assert result.tolist() == [1, 1, 0]


def test_density_per_area():
    assert density_per_area(20, 5) == pytest.approx(40)


def test_density_per_area_rejects_zero():
    with pytest.raises(ValueError):
        density_per_area(10, 0)


def test_within_distance():
    sources = layer([(0, 0)])
    targets = layer([(300, 400), (600, 0)])

    result = within_distance(targets, sources, 500)

    assert len(result) == 1
    assert result.index.tolist() == [0]

