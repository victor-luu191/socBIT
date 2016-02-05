package core;

import java.io.IOException;

import defs.Dataset;
import defs.Hypers;

public class SocBIT {

	static Dataset train_ds;
	static Dataset test_ds;
	
	public static void main(String[] args) throws IOException {
		
		String dataDir = "data/";	// or args[0]
		Dataset ds = loadData(dataDir);
		double train_ratio = 0.8;
		split(ds, train_ratio);
		
		int numTopic = 10;	// can use the pkg org.kohsuke.args4j to obtain named args
		GD_Trainer gd_trainer = init_GD_Trainer(train_ds, numTopic);
		
		Parameters initParams = new Parameters(train_ds.numUser, train_ds.numItem, numTopic, train_ds.numBrand);
		String resDir = "result/";
		Parameters learned_params = gd_trainer.gradDescent(initParams, resDir);
		
		save(learned_params, resDir);
		predict(learned_params, test_ds);
	}
	/**
	 * Make predictions using specified {@link params} and check with ground truth {@link test_ds} to obtain ... evaluation metrics
	 * @param params
	 * @param test_ds
	 */
	private static void predict(Parameters params, Dataset test_ds) {
		// TODO Auto-generated method stub
		
	}

	private static void save(Parameters params, String resDir) {
		// TODO Auto-generated method stub
		
	}

	private static GD_Trainer init_GD_Trainer(Dataset ds, int numTopic) {
		double topicLambda = 10;
		double brandLambda = 100;
		double weightLambda = 100;
		double decisionLambda = 10;
		Hypers hypers = new Hypers(topicLambda, brandLambda, weightLambda, decisionLambda);
		int maxIter = 100;
		GD_Trainer gdTrainer = new GD_Trainer(ds, numTopic, hypers, maxIter);
		return gdTrainer;
	}

	/**
	 * split full ds into {@link training_ds} and {@link test_set}, with {@link training_ds} occupy a ratio {@link train_ratio} 
	 * @param ds
	 * @param train_ratio 
	 */
	private static void split(Dataset ds, double train_ratio) {
		// TODO Auto-generated method stub
		
	}

	private static Dataset loadData(String dir) {
		// TODO Auto-generated method stub
		return null;
	}

}
