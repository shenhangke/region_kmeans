import pyspark
import os
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt

spark = pyspark.sql.SparkSession.builder.master("local[3]"). \
    appName("kmeans").config("spark.some.config.option", "some-value").getOrCreate()


def get_kmeans_result(data_path, class_num):
    data = spark.read.csv(path=data_path, encoding="GBK")
    data = data.filter(r'_c0!="编号"')

    def mapToFeature(line):
        # line["features"] = [(Vectors.dense([float(line["_c3"]), float(line["_c4"])]),)]
        return [(Vectors.dense([float(line["_c3"]), float(line["_c4"])]))]

    data_rdd = data.rdd.map(mapToFeature)
    data_df = spark.createDataFrame(data_rdd, ["features"])
    kmeans = pyspark.ml.clustering.KMeans(k=class_num, seed=1)
    model = kmeans.fit(data_df)

    def mapToTuple(line):
        return ([line["features"]], line["prediction"])

    result_rdd = model.transform(data_df).rdd.map(mapToTuple)
    result_list = result_rdd.collect()
    color = ["red", "blue", "yellow"]
    for (vector, type_name) in result_list:
        array = vector[0].toArray()
        print(array)
        #print(len(vector))
        plt.scatter(array[0],array[1],color=color[type_name])
    plt.show()


if __name__ == "__main__":
    get_kmeans_result(os.path.dirname(__file__) + "/data/滨江区.csv", 3)
