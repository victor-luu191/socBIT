package core;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import defs.Dataset;
import defs.Hypers;

public class SocBIT {

	private static int maxNumUser = 1000;
	private static int maxNumItem = 100000;
	
	static Dataset train_ds;
	static Dataset test_ds;
	
	
	public static void main(String[] args) throws IOException {
		
		String dataDir = "data/syn/";	// or args[0]
		int numBrand = 20;	// temporary value passing, later will read from file data_stats or itemInfo
		Dataset ds = loadData(dataDir, numBrand);
//		double train_ratio = 0.8;
//		split(ds, train_ratio);
		
		int numTopic = 10;	// can use the pkg org.kohsuke.args4j to obtain named args
		
		// currently training on whole data set, switch to training set later
		GD_Trainer gd_trainer = init_GD_Trainer(ds, numTopic);	
		Parameters initParams = new Parameters(ds.numUser, ds.numItem, numTopic, ds.numBrand);
		String resDir = "result/syn/";
		Parameters learned_params = gd_trainer.gradDescent(initParams, resDir);
		
		save(learned_params, resDir);
//		predict(learned_params, test_ds);
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
		System.out.println("Initialized a regularized GD trainer with " + numTopic + " topics.");
		System.out.println("Regularization constants: ");
		System.out.println("topicLambda, brandLambda, weightLambda, decisionLambda" );
		System.out.println(topicLambda + "," + brandLambda + "," + weightLambda + "," + decisionLambda);
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

	private static Dataset loadData(String dir, int numBrand) throws IOException {
		
		RealMatrix ratings = loadRatings(dir + "ratings.csv");
		RealMatrix edge_weights = loadEdgeWeights(dir + "edge_weights.csv");
		return new Dataset(ratings, edge_weights, numBrand);
	}

	// read edge weights from the file and fill in 0s for user pairs with no connection
	private static RealMatrix loadEdgeWeights(String fname) throws NumberFormatException, IOException {
		
		RealMatrix edge_weights = new Array2DRowRealMatrix(maxNumUser, maxNumUser);
		Map<String, Integer> userMap = new HashMap<String, Integer>();
		
		BufferedReader reader = new BufferedReader(new FileReader(fname));
		String line = reader.readLine();	// skip header
		while ((line = reader.readLine()) != null) {
			String[] fields = line.split(",");
			String uid = fields[0];
			String vid = fields[1];
			double weight = Double.valueOf(fields[2]);
			
			int uIndex = lookUpIndex(uid, userMap);
			int vIndex = lookUpIndex(vid, userMap);
			edge_weights.setEntry(uIndex, vIndex, weight);
		}
		reader.close();
		int numUser = userMap.size();
		edge_weights = edge_weights.getSubMatrix(1, numUser, 1, numUser);	// rm redundant rows and cols
		System.out.println("Loaded all edge weights.");
		System.out.println("first row of ede weights");
		System.out.println(edge_weights.getRowVector(0).toString());
		System.out.println();
		return edge_weights;
	}
	private static RealMatrix loadRatings(String fname) throws IOException {
		RealMatrix ratings = new Array2DRowRealMatrix(maxNumUser, maxNumItem);
		
		Map<String, Integer> itemMap = new HashMap<String, Integer>();
		Map<String, Integer> userMap = new HashMap<String, Integer>();
		
		BufferedReader reader = new BufferedReader(new FileReader(fname));
		String line = reader.readLine();	// skip header
		while ((line = reader.readLine()) != null) {
			String[] fields = line.split(",");
			String uid = fields[0];
			String itemId = fields[1];
			double r = Double.valueOf(fields[2]);
			
			int userIndex = lookUpIndex(uid, userMap);
			int itemIndex = lookUpIndex(itemId, itemMap);
			ratings.setEntry(userIndex, itemIndex, r);
		}
		
		reader.close();
		int numUser = userMap.size();
		int numItem = itemMap.size();
		System.out.println("numUser = " + numUser  + ", numItem = " + numItem);
		ratings = ratings.getSubMatrix(1, numUser, 1, numItem);		// rm redundant rows and cols
		System.out.println("Loaded all ratings.");
		System.out.println("First row of ratings: ");
		System.out.println(ratings.getRowVector(0).toString());
		System.out.println();
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
	private static int lookUpIndex(String id, Map<String, Integer> id2Index) {
		
		if (id2Index.containsKey(id)) {
			return id2Index.get(id);
		} else {
			int size = id2Index.size() + 1;
			id2Index.put(id, size);
			return size;
		}
	}

}
