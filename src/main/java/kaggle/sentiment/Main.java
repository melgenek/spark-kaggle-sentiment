package kaggle.sentiment;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SaveMode;

import java.io.IOException;

import static org.apache.spark.sql.functions.col;

public class Main {

	public static void main(String[] args) throws IOException {
		String labeledTrainDataFile = Main.class.getClassLoader().getResource("labeledTrainData.tsv").getPath();
		String testDataFile = Main.class.getClassLoader().getResource("testData.tsv").getPath();
		SparkConf conf = new SparkConf().setMaster("local[*]").setAppName("Sentiment");
		JavaSparkContext sparkContext = new JavaSparkContext(conf);
		SQLContext sqlContext = new SQLContext(sparkContext);

		DataFrameReader dataFrameReader = getDataFrameReader(sqlContext);
		DataFrame labeledTrainData = dataFrameReader.load(labeledTrainDataFile).cache();
		DataFrame testData = dataFrameReader.load(testDataFile).cache();

		//initialize steps for preprocessing
		int featuresCount = 70000;
		ReviewTokenizer reviewTokenizer = new ReviewTokenizer(sqlContext).setInputCol("review").setOutPutCol("uniBiGram");
		HashingTF hashingTF = new HashingTF().setInputCol("uniBiGram").setOutputCol("rawFeatures").setNumFeatures(featuresCount);
		IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
		StringIndexer indexer = new StringIndexer().setInputCol("sentiment").setOutputCol("label");

		LogisticRegression logisticRegression = new LogisticRegression();

		Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{
				reviewTokenizer,
				hashingTF,
				idf,
				indexer,
				logisticRegression
		});

		// Train classifier based on 5-fold cross validation
		ParamMap[] paramGrid = new ParamGridBuilder()
				.addGrid(logisticRegression.regParam(), new double[]{0.001, 0.01, 0.1, 1.0, 5, 10, 100})
				.addGrid(logisticRegression.maxIter(), new int[]{100, 300, 500})
				.build();
		BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator();
		CrossValidator crossValidator = new CrossValidator()
				.setEstimator(pipeline)
				.setEvaluator(evaluator)
				.setEstimatorParamMaps(paramGrid)
				.setNumFolds(5);
		CrossValidatorModel model = crossValidator.fit(labeledTrainData);

		// Estimate test data
		testData = model.transform(testData);

		//Write result
		testData.select(col("id"), col("prediction").as("sentiment")).coalesce(1)
				.write()
				.mode(SaveMode.Overwrite)
				.format("com.databricks.spark.csv")
				.option("header", "true")
				.save("submission");
	}

	private static DataFrameReader getDataFrameReader(SQLContext sqlContext) {
		return sqlContext.read()
				.format("com.databricks.spark.csv")
				.option("delimiter", "\t")
				.option("escape", "\\")
				.option("inferSchema", "true")
				.option("header", "true");
	}

}


