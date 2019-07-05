from math import sin, sqrt, asin, cos

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window


def create_spark_session():
    spark = SparkSession \
        .builder \
        .getOrCreate()
    return spark


def extract_data(spark, input_path):
    """
    extracts data from input file and create data frames.
    :param spark: spark session.
    :param input_path: path of input data file.
    :return: vehicle and operating periods DFs.
    """
    df = spark.read.format('json').load(F'{input_path}2019/06/01/15/*/*.txt')

    vehicles_evts_df = df.filter('on="vehicle"')
    op_prd_evts_df = df.filter('on="operating_period"')

    vehicles_evts_df = vehicles_evts_df.select(
        F.col('data.id').alias('vehicle_id'),
        F.col('data.location.lat').alias('lat'),
        F.col('data.location.lng').alias('lng'),
        'on', 'at', 'event', 'organization_id'
    ).orderBy('vehicle_id')

    op_prd_evts_df = op_prd_evts_df.select(
        F.col('data.id').alias('op_prd_id'),
        F.col('data.start').alias('op_prd_start'),
        F.col('data.finish').alias('op_prd_finish'),
    )

    return vehicles_evts_df, op_prd_evts_df


def calculate_average_distance(vehicles_evts_df, op_prd_evts_df):
    """
    calculate average distance in an operating period of all vehicles and
    per vehicle as well.
    :param vehicles_evts_df: vehicle events DF.
    :param op_prd_evts_df: operating periods DF.
    :return: DF with col `distance`
    """
    # so that we could join both data frames.
    vehicles_evts_df = vehicles_evts_df.withColumn('key', F.lit(1))
    op_prd_evts_df = op_prd_evts_df.withColumn('key', F.lit(1))

    df_merge = vehicles_evts_df.join(op_prd_evts_df, on='key', how='left').drop('key')
    df_merge = df_merge \
        .withColumn('lng', F.toRadians('lng')) \
        .withColumn('lat', F.toRadians('lat'))

    w = Window().partitionBy('op_prd_id', 'vehicle_id').orderBy("at")

    df = df_merge.withColumn('distance', calculate_distance(
        'lng', 'lat',
        F.lag('lng', 1).over(w),
        F.lag('lat', 1).over(w)
    )).alias('distance')

    df = df.withColumn('distance',
                       F.when(F.isnull(df['distance']), 0)
                       .otherwise(df['distance'])
                       ).alias('distance')

    return df


def load_data(df, output_path):
    # average distance per vehicle in an operating period.
    distance_per_vehicle_in_op_prd = df.select('*') \
        .groupBy('op_prd_id', 'vehicle_id') \
        .agg(F.avg('distance').alias('average_distance (kms)')) \
        .orderBy('op_prd_id', 'vehicle_id')

    # average distance of all vehicles in an operating period.
    distance_per_op_prd = df.select('*') \
        .groupBy('op_prd_id') \
        .agg(F.avg('distance').alias('average_distance (kms)'))

    distance_per_vehicle_in_op_prd.show(truncate=False)
    distance_per_op_prd.show(truncate=False)

    distance_per_vehicle_in_op_prd.coalesce(1).write.format("json").partitionBy(['op_prd_id', 'vehicle_id']) \
        .mode("overwrite") \
        .save(f'{output_path}/dist_per_vehicle_in_op_prd')

    distance_per_op_prd.coalesce(1).write.format("json").partitionBy(['op_prd_id']) \
        .mode("overwrite") \
        .save(f'{output_path}/dist_per_op_prd')


@F.udf(FloatType())
def calculate_distance(long_x, lat_x, long_y, lat_y):
    """
    Haversine formula: https://en.wikipedia.org/wiki/Haversine_formula
    :param long_x:
    :param lat_x:
    :param long_y:
    :param lat_y:
    :return:
    """
    if not (long_x and lat_x and long_y and lat_y):
        return 0

    dlon = long_y - long_x if long_y != long_x else long_y
    dlat = lat_y - lat_x if lat_y != lat_x else lat_y

    a = (sin(dlat / 2) ** 2) + cos(lat_x) * cos(lat_y) * (sin(dlon / 2) ** 2)
    c = 2 * asin(sqrt(a))
    r = 6371  # radius of earth in kms.
    return (c * r)


def main():
    input_data_path = "data/"
    output_data_path = "output/"

    spark = create_spark_session()

    # vehicles_evts_df, op_prd_evts_df = extract_data(spark, input_data_path)
    # df = calculate_average_distance(vehicles_evts_df, op_prd_evts_df)
    # load_data(df, output_data_path)


if __name__ == "__main__":
    main()
