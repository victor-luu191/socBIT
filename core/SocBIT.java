package core;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import myUtil.Savers;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import defs.Dataset;
import defs.Errors;
import defs.Hypers;
import defs.InvalidModelException;
import defs.ParamModelMismatchException;

public class SocBIT {

	
	private static int maxNumUser = 2000;
	private static int maxNumItem = 2000;
	private static final int maxDim = maxNumItem;
	
	static Dataset train_ds;
	static Dataset test_ds;
	
	
	public static void main(String[] args) throws IOException, InvalidModelException, ParamModelMismatchException {
		
		// temporary value passing, later will read from file data_stats or itemInfo
		int numUser = 1000;
		int numBrand = 20;	

		// TODO: switch to the pkg org.kohsuke.args4j to enable named args
		String dataDir = "data/syn/N" + numUser + "/unif/";	// or args[0]
		Dataset ds = loadData(dataDir, numBrand);
//		double train_ratio = 0.8;
//		split(ds, train_ratio);
		
		int minK = 5;
		int maxK = 15;
//		String gtParamDir = dataDir + "true_params/";
//		Parameters gt_params = loadParams(gtParamDir);
//		String errStr = "numTopic, topicUserErr, topicItemErr, brandUserErr, brandItemErr, decisionPrefErr \n";
		for (int numTopic = minK; numTopic <=  maxK; numTopic++) {
			Params socBIT_params = trainBySocBIT(ds, numTopic);
			Params ste_params = trainBySTE(ds, numTopic);
//			Errors errors = compDiff(socBIT_params, gt_params);
//			errStr += numTopic + ","  + errors.topicUser + "," + errors.topicItem + "," + errors.brandUser + "," + errors.brandItem + "," + 
//						errors.decisionPrefs + "\n";
		}	
//		String fErrors = "result/syn/N" + ds.numUser + "/unif/" + "/" ;
//		Savers.save(errStr, fErrors);
//		predict(learned_params, test_ds);
	}
	
	private static Params trainBySTE(Dataset ds, int numTopic) throws IOException, InvalidModelException, ParamModelMismatchException {
		
		String resDir = "result/syn/N" + ds.numUser + "/STE/numTopic" + numTopic + "/" ; // "/unif/numTopic" + numTopic + "/"
		if (!Files.exists(Paths.get(resDir))) {
			Files.createDirectories(Paths.get(resDir));
		} 
		
		Trainer trainer = initTrainer("STE", ds, numTopic);	// currently training on whole data set, switch to training set later	
		SocBIT_Params initParams = new SocBIT_Params(ds.numUser, ds.numItem, ds.numBrand, trainer.numTopic);
		Params learned_params = trainer.gradDescent(initParams, resDir);
//		save(learned_params, resDir);
		return learned_params;
	}

	private static Errors compDiff(SocBIT_Params params1, SocBIT_Params params2) {
		
		double topicUserErr = params1.topicUser.subtract(params2.topicUser).getFrobeniusNorm();
		double topicItemErr = params1.topicItem.subtract(params2.topicItem).getFrobeniusNorm();
		double brandUserErr = params1.brandUser.subtract(params2.brandUser).getFrobeniusNorm();
		double brandItemErr = params1.brandItem.subtract(params2.brandItem).getFrobeniusNorm();
		double decisionPrefErr = toVector(params1.userDecisionPrefs).subtract(toVector(params2.userDecisionPrefs)).getNorm();
		
		return new Errors(topicUserErr, topicItemErr, brandUserErr, brandItemErr, decisionPrefErr);
	}

	private static RealVector toVector(double[] arr) {
		return new ArrayRealVector(arr);
	}

	private static Params loadParams(String gtParamsDir) throws IOException {
		System.out.println("Loading gt params from folder " + gtParamsDir);
		double[] decPrefs = loadDecPref(gtParamsDir);
		
		RealMatrix topicUser = loadMat(gtParamsDir, "topic_user_feats.csv");
		RealMatrix brandUser = loadMat(gtParamsDir, "brand_user_feats.csv");
		RealMatrix topicItem = loadMat(gtParamsDir, "topic_item_feats.csv");
		RealMatrix brandItem = loadMat(gtParamsDir, "brand_item_feats.csv");
		
		int numTopic = topicItem.getRowDimension();
		int numUser = topicUser.getColumnDimension();
		int numBrand = brandUser.getRowDimension();
		int numItem = topicItem.getColumnDimension();
		System.out.println("numTopic, numUser, numBrand, numItem");
		System.out.println(numTopic + "," + numUser + "," + numBrand + "," + numItem);
		
		return new SocBIT_Params(decPrefs, topicUser, brandUser, topicItem, brandItem);
	}

	private static RealMatrix loadMat(String gtParamsDir, String fname) throws IOException {
		
		RealMatrix matrix = new Array2DRowRealMatrix(maxDim, maxDim);
		BufferedReader reader = new BufferedReader(new FileReader(gtParamsDir + fname));
		
		int numRow = 0;
		
		String line = reader.readLine();
		int numCol = line.split(",").length - 1;	// bc the first column contains row indices not values
		while ((line = reader.readLine()) != null) {
			numRow ++;
			String[] fields = line.split(",");
			int row = Integer.parseInt(fields[0].replace("\"", ""));
			for (int col = 1; col < fields.length; col++) {
				matrix.setEntry(row, col, Double.parseDouble(fields[col]));
			}
		}
		matrix = matrix.getSubMatrix(1, numRow, 1, numCol);
		reader.close();
		return matrix;
	}

	private static double[] loadDecPref(String gtParamsDir) throws FileNotFoundException, IOException {
		
		String fname = gtParamsDir + "/decision_pref.csv";
		BufferedReader reader = new BufferedReader(new FileReader(fname));
		
		List<Double> decPrefs = new ArrayList<Double>();
		String line = reader.readLine();
		while ((line = reader.readLine()) != null) {
			String[] fields = line.split(",");
//			String uid = fields[0];
			decPrefs.add(Double.parseDouble(fields[1]));
		}
		reader.close();
		double[] prefs = new double[decPrefs.size()];
		for (int u = 0; u < prefs.length; u++) {
			prefs[u] = decPrefs.get(u);
		}
		return prefs;
	}

	private static Params trainBySocBIT(Dataset ds, int numTopic) throws IOException, InvalidModelException, ParamModelMismatchException {
		
		String resDir = "result/syn/N" + ds.numUser + "/socBIT/numTopic" + numTopic + "/" ; // "/unif/numTopic" + numTopic + "/"
		if (!Files.exists(Paths.get(resDir))) {
			Files.createDirectories(Paths.get(resDir));
		} 
		
		Trainer trainer = initTrainer("socBIT", ds, numTopic);	// currently training on whole data set, switch to training set later	
		SocBIT_Params initParams = new SocBIT_Params(ds.numUser, ds.numItem, ds.numBrand, trainer.numTopic);
		Params learned_params = trainer.gradDescent(initParams, resDir);
//		save(learned_params, resDir);
		return learned_params;
	}
	/**
	 * Make predictions using specified {@link params} and check with ground truth {@link test_ds} to obtain ... evaluation metrics
	 * @param params
	 * @param test_ds
	 */
	private static void predict(Params params, Dataset test_ds) {
		// TODO Auto-generated method stub
		
	}

	private static void save(SocBIT_Params params, String resDir) throws IOException {
		
		String userTopicFeat_file = resDir + "user_topic_feats.csv";
		Savers.save(params.topicUser.toString(), userTopicFeat_file);
		
		String userBrandFeat_file = resDir + "user_brand_feats.csv";
		Savers.save(params.brandUser.toString(), userBrandFeat_file);
		
		String itemTopicFeat_file = resDir + "item_topic_feats.csv";
		Savers.save(params.topicItem.toString(), itemTopicFeat_file);
		
		String itemBrandFeat_file = resDir + "item_brand_feats.csv";
		Savers.save(params.brandItem.toString(), itemBrandFeat_file);
		
		String decisionPref_file = resDir + "decision_prefs.csv";
		Savers.save(Arrays.toString(params.userDecisionPrefs), decisionPref_file);
	}

	private static Trainer initTrainer(String model, Dataset ds, int numTopic) throws InvalidModelException {
		
		int maxIter = 100;
		
		Hypers hypers = null;
		if (model.equalsIgnoreCase("socBIT")) {
			hypers = assignHypers4SocBIT();
			System.out.println("Initializing a regularized trainer for SocBIT model. The trainer is assigned " + numTopic + " topics.");
			printRegConst(hypers);
		} 
		else {
			if (model.equalsIgnoreCase("STE")) {
				hypers = assignHypers4STE();
				System.out.println("Initializing a regularized trainer for STE model. The trainer is assigned " + numTopic + " topics.");
			} else {
				throw new InvalidModelException();
			}
		}
		
		Trainer trainer = new Trainer(model, ds, numTopic, hypers, maxIter);
		return trainer;
	}

	private static void printRegConst(Hypers hypers) {
		System.out.println("Regularization constants: ");
		System.out.println("topicLambda, brandLambda, weightLambda, decisionLambda" );
		System.out.println(hypers.topicLambda + "," + hypers.brandLambda + "," + hypers.weightLambda + "," + hypers.decisionLambda);
	}

	private static Hypers assignHypers4STE() {
		double topicLambda = 10;
		double alpha = 0.5;
		// TODO Auto-generated method stub
		return new Hypers(topicLambda, alpha);
	}

	private static Hypers assignHypers4SocBIT() {
		double topicLambda = 10;
		double brandLambda = 100;
		double weightLambda = 100;
		double decisionLambda = 10;
		Hypers hypers = Hypers.setBySocBIT(topicLambda, brandLambda, weightLambda, decisionLambda);
		return hypers;
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
//		System.out.println("first row of ede weights");
//		System.out.println(edge_weights.getRowVector(0).toString());
//		System.out.println();
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
//		System.out.println("First row of ratings: ");
//		System.out.println(ratings.getRowVector(0).toString());
//		System.out.println();
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
