package core;

import helpers.UtilFuncs;

import java.util.Arrays;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class SocBIT_Params extends Params {
	
	public RealMatrix brandUser;	// represent brand interests of users, later need to normalize for each user
	public RealMatrix brandItem;	// represent an item's popularity under each of its producing brand 
									// 	if a brand b does not produce an item i then the entry (i,b) is simply 0	
	
	/**
	 * represent decision preference of users i.e. whether a user prefers brand-based or topic-based adopts
	 * Range: [0,1]; > 0.5 if the user prefers topic-based, < 0.5 if  the user prefers brand-based
	 */
	public double[] userDecisionPrefs;
	
	public SocBIT_Params(double[] userDecisionPrefs, RealMatrix topicUser,
							RealMatrix brandUser, RealMatrix topicItem, RealMatrix brandItem) {
		
		super(topicUser, topicItem);
		
		this.userDecisionPrefs = copyOf(userDecisionPrefs);
		this.brandUser = brandUser.copy();
		this.brandItem = brandItem.copy();
	}

	/**
	 * return default params with correct {@link dimensions}
	 * @param numUser
	 * @param numItem
	 * @param numTopic
	 * @param numBrand
	 */
	public SocBIT_Params(int numUser, int numItem, int numBrand, int numTopic) {
		
		super(numUser, numItem, numTopic);
		initUserTopicFeats(numUser, numTopic);
		initItemTopicFeats(numItem, numTopic);

		initUserBrandFeats(numUser, numBrand);
		initItemBrandFeats(numItem, numBrand);
		
		userDecisionPrefs = new double[numUser];
		// as we expect that most users are neutral, neither brand-based nor topic-based extreme, 
		// we initialize all decision prefs as 0.5
		Arrays.fill(userDecisionPrefs, 0.5);	
	}

	public SocBIT_Params(SocBIT_Params params) {
		
		super(params.topicUser, params.topicItem);
		
		userDecisionPrefs = copyOf(params.userDecisionPrefs);
		brandUser = params.brandUser.copy();
		brandItem = params.brandItem.copy();
	}

	private void initItemBrandFeats(int numItem, int numBrand) {
		
		brandItem = new Array2DRowRealMatrix(numBrand, numItem);
		for (int i = 0; i < numItem; i++) {
			brandItem.setColumnVector(i, uniformVector(numBrand));	 
		}
	}

	private void initUserBrandFeats(int numUser, int numBrand) {
		
		brandUser = new Array2DRowRealMatrix(numBrand, numUser);
		for (int u = 0; u < numUser; u++) {
			brandUser.setColumnVector(u, uniformVector(numBrand));	 
		}
	}

	double sqDiff(SocBIT_Params other) {
		double topicDiff = this.topicDiff(other);
		double brandDiff = this.brandDiff(other);
		double decisionDiff = this.decisionDiff(other);
		double sqDiff = topicDiff + brandDiff + decisionDiff;
		return sqDiff;
	}

	private double decisionDiff(SocBIT_Params other) {
		double decisionDiff = 0;
		
		int numUser = this.topicUser.getColumnDimension();
		for (int u = 0; u < numUser; u++) {
			decisionDiff += UtilFuncs.square(this.userDecisionPrefs[u] - other.userDecisionPrefs[u]);
		}
		return decisionDiff;
	}

	private double brandDiff(SocBIT_Params other) {
		double brandDiff = UtilFuncs.square(this.brandUser.subtract(other.brandUser).getFrobeniusNorm());
		brandDiff += UtilFuncs.square(this.brandItem.subtract(other.brandItem).getFrobeniusNorm());
		return brandDiff;
	}
	
	private double[] copyOf(double[] arr) {
		return Arrays.copyOf(arr, arr.length);
	}
}
