package core;

import java.util.Arrays;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Parameters {
	
	private static final double EPSILON = Math.pow(10, -2);

	/**
	 * represent decision preference of users i.e. whether a user prefers brand-based or topic-based adopts
	 * Range: [0,1]; > 0.5 if the user prefers topic-based, < 0.5 if  the user prefers brand-based
	 */
	double[] userDecisionPrefs;	
	
	RealMatrix topicUser;	// represent topic interests of users, later need to normalize for each user
	RealMatrix brandUser;	// represent brand interests of users, later need to normalize for each user
	
	RealMatrix topicItem;	// represent relevance of each item with different topics, later need to normalize for each topic

	// represent popularity of each item under each brand producing the item; 
	// if a brand b does not produce an item i then the entry (i,b) is simply 0 
	RealMatrix brandItem;	
	
	/**
	 * return default params with correct {@link dimensions}
	 * @param numUser
	 * @param numItem
	 * @param numTopic
	 * @param numBrand
	 */
	public Parameters(int numUser, int numItem, int numTopic, int numBrand) {
		
		initUserTopicFeats(numUser, numTopic);
		initUserBrandFeats(numUser, numBrand);
		initItemTopicFeats(numItem, numTopic);
		initItemBrandFeats(numItem, numBrand);
		
		userDecisionPrefs = new double[numUser];
		// as we expect that most users are neutral, neither brand-based nor topic-based extreme, 
		// we initialize all decision prefs as 0.5
		Arrays.fill(userDecisionPrefs, 0.5);	
	}

	private void initItemBrandFeats(int numItem, int numBrand) {
		brandItem = new Array2DRowRealMatrix(numBrand, numItem);
		RealVector unitVector = unitVector(numBrand);
		RealVector smallVector = unitVector.mapMultiply(EPSILON);
		for (int i = 0; i < numItem; i++) {
			brandItem.setColumnVector(i, smallVector);	// unitVector
		}
	}

	private void initItemTopicFeats(int numItem, int numTopic) {
		topicItem = new Array2DRowRealMatrix(numTopic, numItem);
		RealVector unitVector = unitVector(numTopic);
		RealVector smallVector = unitVector.mapMultiply(EPSILON);
		for (int i = 0; i < numItem; i++) {
			topicItem.setColumnVector(i, smallVector);	// unitVector
		}
	}

	private void initUserBrandFeats(int numUser, int numBrand) {
		brandUser = new Array2DRowRealMatrix(numBrand, numUser);
		RealVector unitVector = unitVector(numBrand);
		RealVector smallVector = unitVector.mapMultiply(EPSILON);
		for (int u = 0; u < numUser; u++) {
			brandUser.setColumnVector(u, smallVector);	// unitVector
		}
	}

	private void initUserTopicFeats(int numUser, int numTopic) {
		topicUser = new Array2DRowRealMatrix(numTopic, numUser);
		RealVector unitVector = unitVector(numTopic);
		RealVector smallVector = unitVector.mapMultiply(EPSILON);
		for (int u = 0; u < numUser; u++) {
			topicUser.setColumnVector(u, smallVector);	// unitVector
		}
	}

	private RealVector unitVector(int size) {
		RealVector unitVector = new ArrayRealVector(size);
		unitVector.setEntry(0, 1);
		for (int k = 1; k < size; k++) {
			unitVector.setEntry(k, 0);
		}
		return unitVector;
	}

	public Parameters(Parameters params) {
		
		userDecisionPrefs = params.userDecisionPrefs;
		topicUser = params.topicUser;
		brandUser = params.brandUser;
		
		topicItem = params.topicItem;
		brandItem = params.brandItem;
	}
}
