package core;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

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

	private static Dataset loadData(String dir) throws IOException {
		
		RealMatrix ratings = loadRatings(dir + "ratings.csv");
		RealMatrix edge_weights = loadEdgeWeights(dir + "edge_weights.csv");
		return new Dataset(ratings, edge_weights);
	}

	// read edge weights from the file and fill in 0s for user pairs with no connection
	private static RealMatrix loadEdgeWeights(String fname) {
		// TODO Auto-generated method stub
		return null;
	}
	private static RealMatrix loadRatings(String fname) throws IOException {
		// TODO Auto-generated method stub
		int maxNumUser = 1000;
		int maxNumItem = 100000;
		RealMatrix ratings = new Array2DRowRealMatrix(maxNumUser, maxNumItem);
		
		int numItem = 0;
		Map<String, Integer> itemMap = new HashMap<String, Integer>();
		int numUser = 0;
		Map<String, Integer> userMap = new HashMap<String, Integer>();
		
		BufferedReader reader = new BufferedReader(new FileReader(fname));
		
		String line;
		while ((line = reader.readLine()) != null) {
			String[] fields = line.split(",");
			String uid = fields[0];
			String itemId = fields[1];
			double r = Double.valueOf(fields[2]);
			
			int userIndex = lookUpIndex(uid, userMap, numUser);
			int itemIndex = lookUpIndex(itemId, itemMap, numItem);
			ratings.setEntry(userIndex, itemIndex, r);
		}
		
		reader.close();
		ratings = ratings.getSubMatrix(1, numUser, 1, numItem);
		return ratings;
	}
	
	/**
	 * Look up the index corresponding to the given {@link id} if {@link id} can be found in the map {@link id2Index}, 
	 * Otherwise add the {@link id} to the map and increment the map {@link size} 
	 * @param id
	 * @param id2Index
	 * @param size
	 * @return
	 */
	private static int lookUpIndex(String id, Map<String, Integer> id2Index, int size) {
		
		if (id2Index.containsKey(id)) {
			return id2Index.get(id);
		} else {
			size ++;
			id2Index.put(id, size);
			return size;
		}
	}

}
