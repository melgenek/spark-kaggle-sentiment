package kaggle.sentiment;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.*;

import java.util.UUID;

import static kaggle.sentiment.Functions.registerReviewToTokensFunction;
import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

// Pipeline stage for cleaning reviews
public class ReviewTokenizer extends Transformer {

	public static final String REVIEW_TO_TOKENS = "reviewToTokens";

	private String uid;
	private Param<String> inputCol;
	private Param<String> outputCol;

	public ReviewTokenizer() {
		this(getUid());
	}

	public ReviewTokenizer(String uid) {
		this.uid = uid;
		inputCol = new Param<>(this, "inputCol", "input column name");
		outputCol = new Param<>(this, "outputCol", "output column name");
		setDefault(inputCol, getUid() + "__input");
		setDefault(outputCol, getUid() + "__output");
	}

	public ReviewTokenizer(SQLContext sqlContext) {
		this();
		registerReviewToTokensFunction(sqlContext);
	}

	@Override
	public StructType transformSchema(StructType schema) {
		return schema.add(new StructField(getOrDefault(outputCol),
				DataTypes.createArrayType(DataTypes.StringType), false, Metadata.empty()));
	}

	@Override
	public DataFrame transform(DataFrame dataset) {
		return dataset.withColumn(getOrDefault(outputCol), callUDF(REVIEW_TO_TOKENS, col(getOrDefault(inputCol))));
	}

	@Override
	public Transformer copy(ParamMap extra) {
		return defaultCopy(extra);
	}

	@Override
	public String uid() {
		return uid;
	}

	private static String getUid() {
		String s = UUID.randomUUID().toString();
		return "ReviewTokenizer_" + s.substring(s.length() - 12);
	}

	public Param<String> getInputCol() {
		return inputCol;
	}

	public Param<String> getOutputCol() {
		return outputCol;
	}


	public ReviewTokenizer setInputCol(String value) {
		set(inputCol, value);
		return this;
	}

	public ReviewTokenizer setOutPutCol(String value) {
		set(outputCol, value);
		return this;
	}

}
