package kaggle.sentiment;

import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataTypes;

import java.util.List;
import java.util.UUID;

public class Functions {

	public static void registerReviewToTokensFunction(SQLContext sqlContext) {
		UDF1<String, List<String>> cleanReviewFunction = (UDF1<String, List<String>>) ReviewCleaner::reviewToTokens;
		ArrayType stringArray = DataTypes.createArrayType(DataTypes.StringType);
		sqlContext.udf().register("reviewToTokens", cleanReviewFunction, stringArray);
	}

}
