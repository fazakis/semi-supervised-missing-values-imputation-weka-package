/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * IRMI.java
 * Copyright (C) 2016 University of Waikato, Hamilton, NZ
 */

package weka.filters.unsupervised.attribute.missingvaluesimputation;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.ParallelMultipleRegressor;
import weka.classifiers.trees.RandomForest;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

/**
 <!-- globalinfo-start -->
 * Uses the IRSSI algorithm as published by Fazakis et al in 'Iterative Robust Semi-Supervised based Imputation (IRSSI) for Missing Data'.<br>
 * <br>
 * Nikos Fazakis, Georgios Kostopoulos, Sotiris Kotsiantis, Iosif Mporas (2019), Iterative Robust Semi-Supervised based Imputation (IRSSI) for Missing Data, Submitted IEEE ACCESS.
 * <br><br>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Templ2011,
 *    author = {Nikos Fazakis and Georgios Kostopoulos and Sotiris Kotsiantis and Iosif Mporas},
 *    journal = {Submitted IEEE ACCESS},
 *    number = {XX},
 *    pages = {XXXX-XXXX},
 *    title = {Iterative Robust Semi-Supervised based Imputation (IRSSI) for Missing Data},
 *    volume = {XX},
 *    year = {2019},
 *    ISSN = {XXXX-XXXX},
 *    HTTP = {XXXX}
 * }
 * </pre>
 * <br><br>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p>
 * 
 * <pre> -nominal-classifier &lt;classname + options&gt;
 *  Nominal classifier to use
 *  (default: : No model built yet.)</pre>
 * 
 * <pre> -numeric-classifier &lt;classname + options&gt;
 *  Nominal classifier to use
 *  (default: Linear Regression: No model built yet.)</pre>
 * 
 * <pre> -num-epochs &lt;int&gt;
 *  Max number of epochs
 *  (default: 100)</pre>
 * 
 * <pre> -epsilon &lt;double&gt;
 *  Epsilon for early termination
 *  (default: 5.0)</pre>
 * 
 <!-- options-end -->
 *
 * @author Nikos Fazakis
 * @version $Revision$
 */
public class IRSSI
extends AbstractImputation
implements TechnicalInformationHandler {
	
	static final long serialVersionUID = 4861140871932161878L;

	/** Array for storing the generated base classifiers. */
	public Classifier[] mreg_Classifiers = {
			new weka.classifiers.functions.LinearRegression(),
			new weka.classifiers.trees.M5P(),
			new weka.classifiers.trees.RandomForest()
	};
	
	/** the flag for the nominal classifier. */
	public final static String NOMINAL_CLASSIFIER = "nominal-classifier";

	/** the flag for the numeric classifier. */
	public final static String NUMERIC_CLASSIFIER = "numeric-classifier";

	/** the flag for the number of epochs. */
	public final static String NUM_EPOCHS = "num-epochs";

	/** the flag for the early termination value. */
	public final static String EPSILON = "epsilon";

	/** the classifier for nominal attributes. */
	protected Classifier m_nominalClassifier = getDefaultNominalClassifier();

	/** the number of epochs. */
	protected int m_numEpochs = getDefaultNumEpochs();

	/** the early termination value. */
	protected double m_epsilon = getDefaultEpsilon();

	/** the classifiers used internally. */
	protected Classifier[] m_classifiers;

	/** the temporary header. */
	protected Instances m_Header;

	/**
	 * Returns general information on the algorithm.
	 *
	 * @return the information on the algorithm
	 */
	@Override
	public String globalInfo() {
		return
				"Uses the IRSSI algorithm as published by Fazakis et al in 'Iterative "
				+ "Robust Semi-Supervised based Imputation (IRSSI) for Missing Data'.\n"
				+ "\n"
				+ getTechnicalInformation();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 *
	 * @return 		the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation 	result;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "Nikos Fazakis and Georgios Kostopoulos and Sotiris Kotsiantis and Iosif Mporas");
		result.setValue(Field.TITLE, "Iterative Robust Semi-Supervised based Imputation (IRSSI) for Missing Data");
		result.setValue(Field.JOURNAL, "Submitted IEEE ACCESS");
		result.setValue(Field.YEAR, "2019");
		result.setValue(Field.VOLUME, "XX");
		result.setValue(Field.NUMBER, "XX");
		result.setValue(Field.PAGES, "XXXX-XXXX");
		result.setValue(Field.ISSN, "XXXX-XXXX");
		result.setValue(Field.HTTP, "XXXX");

		return result;
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> result = new Vector<Option>();
		
		
		
		result.addElement(new Option(
		          "\tFull class name of classifier to include, followed\n"
		                  + "\tby scheme options. May be specified multiple times.\n"
		                  + "\t(default: \"weka.classifiers.rules.ZeroR\")",
		                  "B", 1, "-B <classifier specification>"));
		
		//result.addAll(Collections.list(super.listOptions()));
		
		
		

		result.addElement(new Option(
				"\t" + nominalClassifierTipText() + "\n"
						+ "\t(default: " + getDefaultNominalClassifier() + ")",
						NOMINAL_CLASSIFIER, 1, "-" + NOMINAL_CLASSIFIER + " <classname + options>"));

//		result.addElement(new Option(
//				"\t" + nominalClassifierTipText() + "\n"
//						+ "\t(default: " + getDefaultNumericClassifier() + ")",
//						NUMERIC_CLASSIFIER, 1, "-" + NUMERIC_CLASSIFIER + " <classname + options>"));

		result.addElement(new Option(
				"\t" + numEpochsTipText() + "\n"
						+ "\t(default: " + getDefaultNumEpochs() + ")",
						NUM_EPOCHS, 1, "-" + NUM_EPOCHS + " <int>"));

		result.addElement(new Option(
				"\t" + epsilonTipText() + "\n"
						+ "\t(default: " + getDefaultEpsilon() + ")",
						EPSILON, 1, "-" + EPSILON + " <double>"));

		result.addAll(Collections.list(super.listOptions()));
		
		
		
		
		
		for (Classifier classifier : getClassifiers()) {
		      if (classifier instanceof OptionHandler) {
		        result.addElement(new Option(
		          "",
		          "", 0, "\nOptions specific to classifier "
		            + classifier.getClass().getName() + ":"));
		        result.addAll(Collections.list(((OptionHandler)classifier).listOptions()));
		      }
		    }
		

		return result.elements();
	}
	
	
	
	
	/**
	   * Gets a single classifier from the set of available classifiers.
	   *
	   * @param index the index of the classifier wanted
	   * @return the Classifier
	   */
	  public Classifier getClassifier(int index) {

	    return mreg_Classifiers[index];
	  }
	
	/**
	   * Gets the classifier specification string, which contains the class name of
	   * the classifier and any options to the classifier
	   *
	   * @param index the index of the classifier string to retrieve, starting from
	   * 0.
	   * @return the classifier string, or the empty string if no classifier
	   * has been assigned (or the index given is out of range).
	   */
	  protected String getClassifierSpec(int index) {

	    if (mreg_Classifiers.length < index) {
	      return "";
	    }
	    Classifier c = getClassifier(index);
	    return c.getClass().getName() + " "
	      + Utils.joinOptions(((OptionHandler)c).getOptions());
	  }
	
	

	/**
	 * Gets the current settings of the filter.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	@Override
	public String[] getOptions() {
		List<String> result = new ArrayList<String>();
		
		
		
		
		for (int i = 0; i < mreg_Classifiers.length; i++) {
		      result.add("-B");
		      result.add("" + getClassifierSpec(i));
		    }
		
		
		

		result.add("-" + NOMINAL_CLASSIFIER);
		result.add(Utils.toCommandLine(m_nominalClassifier));

//		result.add("-" + NUMERIC_CLASSIFIER);
//		result.add(Utils.toCommandLine(m_numericClassifier));

		result.add("-" + NUM_EPOCHS);
		result.add("" + m_numEpochs);

		result.add("-" + EPSILON);
		result.add("" + m_epsilon);

		Collections.addAll(result, super.getOptions());

		return result.toArray(new String[result.size()]);
	}

	/**
	 * Parses a given list of options.
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	@Override
	public void setOptions(String[] options) throws Exception {
		
		
		// Iterate through the schemes
	    Vector<Classifier> classifiers = new Vector<Classifier>();
	    while (true) {
	      String classifierString = Utils.getOption('B', options);
	      if (classifierString.length() == 0) {
	        break;
	      }
	      String [] classifierSpec = Utils.splitOptions(classifierString);
	      if (classifierSpec.length == 0) {
	        throw new IllegalArgumentException("Invalid classifier specification string");
	      }
	      String classifierName = classifierSpec[0];
	      classifierSpec[0] = "";
	      classifiers.addElement(AbstractClassifier.forName(classifierName,
	            classifierSpec));
	    }
	    if (classifiers.size() == 0) {
	      classifiers.addElement(new weka.classifiers.rules.ZeroR());
	    }
	    Classifier [] classifiersArray = new Classifier [classifiers.size()];
	    for (int i = 0; i < classifiersArray.length; i++) {
	      classifiersArray[i] = (Classifier) classifiers.elementAt(i);
	    }
	    setClassifiers(classifiersArray);
		
		
		
		
		
		String 	tmpStr;
		String[]	tmpOptions;

		tmpStr = Utils.getOption(NOMINAL_CLASSIFIER, options);
		if (!tmpStr.isEmpty()) {
			tmpOptions    = Utils.splitOptions(tmpStr);
			tmpStr        = tmpOptions[0];
			tmpOptions[0] = "";
			setNominalClassifier((Classifier) Utils.forName(Classifier.class, tmpStr, tmpOptions));
		}
		else {
			setNominalClassifier(getDefaultNominalClassifier());
		}

//		tmpStr = Utils.getOption(NUMERIC_CLASSIFIER, options);
//		if (!tmpStr.isEmpty()) {
//			tmpOptions    = Utils.splitOptions(tmpStr);
//			tmpStr        = tmpOptions[0];
//			tmpOptions[0] = "";
//			setNumericClassifier((Classifier) Utils.forName(Classifier.class, tmpStr, tmpOptions));
//		}
//		else {
//			setNumericClassifier(getDefaultNumericClassifier());
//		}

		tmpStr = Utils.getOption(NUM_EPOCHS, options);
		if (!tmpStr.isEmpty())
			setNumEpochs(Integer.parseInt(tmpStr));
		else
			setNumEpochs(getDefaultNumEpochs());

		tmpStr = Utils.getOption(EPSILON, options);
		if (!tmpStr.isEmpty())
			setEpsilon(Double.parseDouble(tmpStr));
		else
			setEpsilon(getDefaultEpsilon());

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Returns the default nominal classifier.
	 *
	 * @return the default
	 */
	protected Classifier getDefaultNominalClassifier() {
		return new RandomForest();
	}

	/**
	 * Sets the classifier to use for nominal attributes.
	 *
	 * @param value the classifier
	 */
	public void setNominalClassifier(Classifier value) {
		m_nominalClassifier = value;
	}

	/**
	 * Returns the classifier in use for nominal attributes.
	 *
	 * @return the classifier
	 */
	public Classifier getNominalClassifier() {
		return m_nominalClassifier;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String nominalClassifierTipText() {
		return "Nominal classifier to use";
	}
	
	
	
	
	
	/**
	   * Returns the tip text for this property
	   * @return tip text for this property suitable for
	   * displaying in the explorer/experimenter gui
	   */
	  public String classifiersTipText() {
	    return "The base regression classifiers to be used.";
	  }

	/**
	 * Sets the list of possible classifers to choose from.
	 *
	 * @param classifiers an array of classifiers with all options set.
	 */
	public void setClassifiers(Classifier [] classifiers) {		
		mreg_Classifiers = classifiers;
	}

	/**
	 * Gets the list of possible classifers to choose from.
	 *
	 * @return the array of Classifiers
	 */
	public Classifier [] getClassifiers() {

		return mreg_Classifiers;
	}
	
	
	
	
	
	

//	protected Classifier getDefaultNumericClassifier() {
//		return new LinearRegression();
//	}
//
//	/**
//	 * Sets the classifier to use for numeric attributes.
//	 *
//	 * @param value the classifier
//	 */
//	public void setNumericClassifier(Classifier value) {
//		m_numericClassifier = value;
//	}

//	/**
//	 * Returns the classifier in use for numeric attributes.
//	 *
//	 * @return the classifier
//	 */
//	public Classifier getNumericClassifier() {
//		return m_numericClassifier;
//	}

//	/**
//	 * Returns the tip text for this property
//	 *
//	 * @return tip text for this property suitable for displaying in the
//	 *         explorer/experimenter gui
//	 */
//	public String numericClassifierTipText() {
//		return "Numeric classifier to use";
//	}

	/**
	 * Returns the default number of epochs.
	 *
	 * @return the default
	 */
	protected int getDefaultNumEpochs() {
		return 10;
	}

	/**
	 * Sets the number of epochs.
	 *
	 * @param value the epochs
	 */
	public void setNumEpochs(int value) {
		m_numEpochs = value;
	}

	/**
	 * Returns the number of epochs.
	 *
	 * @return the epochs
	 */
	public int getNumEpochs() {
		return m_numEpochs;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String numEpochsTipText() {
		return "Max number of epochs";
	}

	/**
	 * Returns the default value for the early termination value.
	 *
	 * @return the default
	 */
	protected double getDefaultEpsilon() {
		return 5;
	}

	/**
	 * Sets the early termination value.
	 *
	 * @param value the termination value
	 */
	public void setEpsilon(double value) {
		m_epsilon = value;
	}

	/**
	 * Returns the early termination value.
	 *
	 * @return the termination value
	 */
	public double getEpsilon() {
		return m_epsilon;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String epsilonTipText() {
		return "Epsilon for early termination";
	}

	/**
	 * Returns the Capabilities of this filter.
	 *
	 * @return            the capabilities of this object
	 * @see               Capabilities
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enableAllClasses();
		result.enable(Capability.MISSING_CLASS_VALUES);
		result.enable(Capability.NO_CLASS);

		return result;
	}

	/**
	 * Computes the median.
	 *
	 * @param vals	the values to compute the median for
	 * @return		the median
	 */
	protected double median(double[] vals) {
		if (vals.length == 0)
			return 0.0;

		ArrayList<Double> newVals = new ArrayList<Double>(vals.length);
		for (double val : vals) {
			if (!Utils.isMissingValue(val))
				newVals.add(val);
		}

		if (newVals.size() == 0)
			return 0.0;

		Collections.sort(newVals);

		// if the array is even, get the avg of the middle two
		int midPoint = newVals.size() / 2;
		if (newVals.size() % 2 == 0)
			return (newVals.get(midPoint) + newVals.get(midPoint-1)) / 2.0;
		else
			return newVals.get(midPoint);
	}

	/**
	 * Computes the mode.
	 *
	 * @param vals	the label indices to compute the mode for
	 * @return		the mode
	 */
	protected double mode(double[] vals) {
		if (vals.length == 0)
			return 0.0;

		Map<Double, Integer> counts = new HashMap<Double, Integer>();
		for (double num : vals) {
			if (Utils.isMissingValue(num))
				continue;
			if (counts.get(num) == null)
				counts.put(num, 1);
			else
				counts.put(num, counts.get(num) + 1);
		}

		double bestKey = 0;
		double bestVal = Double.NEGATIVE_INFINITY;
		Set<Double> keys = counts.keySet();
		for (double key : keys) {
			if (counts.get(key) > bestVal) {
				bestKey = key;
				bestVal = counts.get(key);
			}
		}

		return bestKey;
	}

	/**
	 * Performs the actual initialization of the imputation algorithm with the
	 * training data.
	 *
	 * @param data the training data
	 * @return the potentially modified header
	 * @throws Exception if the training fails
	 */
	@Override
	protected Instances doBuildImputation(Instances data) throws Exception {
		System.out.println("doBuildImputation Has RUN!");
		Instances df = new Instances(data);
		int originalClassIndex = df.classIndex();

		for (int i = 0; i < df.numInstances(); i++)
			df.get(i).setClassValue(Double.NaN);

		/*
		 * Step 2: Sort the variables according to the original amount of
		 * missing values. We now assume that the variables are already sorted,
		 * i.e. M(x1) >= ... >= M(x2), where M(xj) denotes the number of missing
		 * cells in variable xj. Set I = {1..p}
		 */

		/*
		 * Step 4 prelim: Create a list of missing/observed indices
		 * with respect to each potential attribute.
		 */
		Pair[] numMissing = new Pair[df.numAttributes()];
		ArrayList<ArrayList<Integer>> missingIndices = new ArrayList<ArrayList<Integer>>();
		ArrayList<ArrayList<Integer>> observedIndices = new ArrayList<ArrayList<Integer>>();
		for (int l = 0; l < df.numAttributes(); l++) {
			int missingCount = 0;
			ArrayList<Integer> missing = new ArrayList<Integer>();
			ArrayList<Integer> observed = new ArrayList<Integer>();
			for (int i = 0; i < df.numInstances(); i++) {
				if (df.get(i).isMissing(l)) {
					missing.add(i);
					missingCount += 1;
				}
				else {
					observed.add(i);
				}
			}
			missingIndices.add(missing);
			observedIndices.add(observed);
			numMissing[l] = new Pair(missingCount, l);
		}
		Arrays.sort(numMissing);

		/*
		 * Step 1: Initialise the missing values using a simple imputation
		 * technique (e.g. k-nearest neighbour or mean imputation).
		 */
		for (int i = 0; i < df.numAttributes(); i++) {
			if (i == df.classIndex())
				continue;

			double[] vals = df.attributeToDoubleArray(i);
			double colMean;
			if (df.attribute(i).isNumeric())
				colMean = median(vals);
			else // it is nominal
				colMean = mode(vals);

			for (int y = 0; y < vals.length; y++) {
				if (Double.isNaN(df.get(y).value(i)) || Double.isInfinite(df.get(y).value(i)))
					df.get(y).setValue(i, colMean);
			}
		}
		boolean[] isStable = new boolean[df.numAttributes()];
		for (int i = 0; i < df.numAttributes(); i++)
			isStable[i] = false;

		/*
		 * Step 4: Create a matrix with the variables corresponding to
		 * the observed and missing cells of x_l.
		 */
		m_classifiers = new Classifier[df.numAttributes()];

		/*
		 * Step 3: Set l = 0 (i.e. l = 1)
		 */
		for (int epochs = 0; epochs < getNumEpochs(); epochs++) {
			for (Pair p : numMissing) {
				int l = p.index;
				if (p.value == 0) // none missing
					continue;
				if (l == originalClassIndex)
					continue;
				if (isStable[l])
					continue;
				if (observedIndices.get(l).size() == 0)
					continue;

				Instances observed = new Instances(data, 0);
				for (int i : observedIndices.get(l))
					observed.add(df.get(i));
				observed.setClassIndex(l);

				Classifier cls;
				if (df.attribute(l).isNominal())
					cls = AbstractClassifier.makeCopy(m_nominalClassifier);
				else{					
					cls = new ParallelMultipleRegressor(mreg_Classifiers);					
				}
				
				
				

				/*
				 * Step 5: Build the model. Then predict the missing class using
				 * semi-supervised techniques.
				 */
				
				m_classifiers[l] = cls;
				double sumOfSquares = 0;
				df.setClassIndex(l);
				
				Instances labeled = new Instances(observed);
				Instances unlabeled = new Instances(observed,0);
				//System.out.println(Arrays.toString(missingIndices.get(l).toArray()));
				
				//Save the positions of the missing value instances
				ArrayList<Integer> missingTruePos = new ArrayList<Integer>();				
				for (int i : missingIndices.get(l)) { //missings per attribute l
					unlabeled.add(df.get(i));
					missingTruePos.add(i);					
				}				
				
				double maxPercentUnlabeledFromUnlabeled=0.1;
				int maxIter = 10;
				int numOfBestUnlabeled = new Double(
						maxPercentUnlabeledFromUnlabeled*unlabeled.numInstances()).intValue();
				if(numOfBestUnlabeled==0)
					numOfBestUnlabeled=1;
				
				System.out.println("\n\nnumOfBestUnlabeled: "+numOfBestUnlabeled);
				
				//SEMI LOOP
				if (df.attribute(l).isNominal()) {//classification
					
					
					System.out.println("\nClassification Selftrain Start");
					
					
					int numberOfClass = df.attribute(l).numValues();
					for(int i=0;i<maxIter && unlabeled.size()>0; i++){						
						Double [][] pre = new Double[unlabeled.size()][4];//maxProb,Position,Prediction,MissingIndex
						double [][] probabilities = new double[unlabeled.size()][numberOfClass];
						
						cls.buildClassifier(labeled);
						
						
						
						for (int j = 0; j < unlabeled.numInstances(); j++) {
							pre[j][1] = (double)j;
							pre[j][2] = cls.classifyInstance(unlabeled.instance(j));			
							pre[j][3] = (double)missingTruePos.get(j);
							if(numberOfClass>0)
								try{//get probabilities & stop if error
									probabilities[j] = cls.distributionForInstance(unlabeled.instance(j));
									//System.out.println(Arrays.toString(probabilities[j]));
									
									double maxProb=0.0;
									for(int m=0;m<numberOfClass;m++)
										if(maxProb<probabilities[j][m])
											maxProb=probabilities[j][m];
									pre[j][0]=maxProb;
									//System.out.println("Max Prob: "+pre[j][0]);
									
								} catch(Exception ex){
									System.out.println("\nError "+ex.getMessage());
									System.exit(1);
								}					 	
						}
						
						Arrays.sort(pre,new Comparator<Double[]>(){
							@Override
							public int compare(final Double[] entry1,final Double[] entry2){
								final Double range1 = entry1[0];
								final Double range2 = entry2[0];
								return range2.compareTo(range1);
							}
						});						
						
						// Add best unlabeled to labeled and after remove them (there's a reason for 2 steps)
						int maxBestUnlabeled = Math.min(pre.length, numOfBestUnlabeled); // unlabeled must be enough
						Integer[] pointerDeleteUnlabeled = new Integer[maxBestUnlabeled];
						//System.out.println("Added: "+maxBestUnlabeled);
						for(int k=0;k<maxBestUnlabeled;k++){
							//Use Missing Index
							double currentClassValue = df.get(new Double(pre[k][3]).intValue()).value(l);
							double newClassValue = pre[k][2];
							df.get(new Double(pre[k][3]).intValue()).setValue(l, newClassValue);
							sumOfSquares += Math.pow(currentClassValue - newClassValue, 2);
							
							//Use unlabeled index
							//System.out.println("aa"+Arrays.deepToString(predAv[k]));						
							unlabeled.instance(new Double(pre[k][1]).intValue()).setClassValue(pre[k][2]);
							labeled.add(unlabeled.instance(new Double(pre[k][1]).intValue()));						
							pointerDeleteUnlabeled[k]=new Double(pre[k][1]).intValue();
						}			
						// it's important to iterate from last to first, because when we remove
						// an instance, the rest shifts by one position.
						Arrays.sort(pointerDeleteUnlabeled);
						//System.out.println(Arrays.deepToString(pointerDeleteUnlabeled));
						for(int k=maxBestUnlabeled-1;k>=0;k--) {
							//Check code
							//They both are in the same index
							/*
							 * System.out.println("\n\n--Start----");
							 * System.out.println(unlabeled.get(pointerDeleteUnlabeled[k]));
							 * System.out.println(df.get(missingTruePos.get(pointerDeleteUnlabeled[k])));
							 * System.out.println("\n\n--End----");
							 */
							unlabeled.delete(pointerDeleteUnlabeled[k]);
							missingTruePos.remove((int)pointerDeleteUnlabeled[k]);
							
						}
						
						System.out.println("Labeled: "+labeled.size()+" Unalebled: "+unlabeled.size());
						
					}//for loop
					
				
				cls.buildClassifier(labeled); //augmented labeled
				
				System.out.println("Classification Selftrain OUTSIDE");
					
					
					
				}else {//regresion					
					
					System.out.println("\nRegression Selftrain Start");
					
					ParallelMultipleRegressor mRegs = (ParallelMultipleRegressor)cls;
					mRegs.setNumExecutionSlots(3);
					if(mRegs.getClassifiers().length!=3)
						throw new Exception("Number of base classifiers should be 3 explicitly.");					
					
					for(int i=0;i<maxIter && unlabeled.size()>0; i++){						
						mRegs.buildClassifier(labeled);
						
						//save predictions
						Double[][] pred = new Double[unlabeled.numInstances()][3];
						Double[][] predAv = new Double[unlabeled.numInstances()][4];						
						for(int j=0;j<unlabeled.numInstances();j++){			
							pred[j][0] = mRegs.getClassifier(0).classifyInstance(unlabeled.instance(j));
							pred[j][1] = mRegs.getClassifier(1).classifyInstance(unlabeled.instance(j));
							pred[j][2] = mRegs.getClassifier(2).classifyInstance(unlabeled.instance(j));
							
							double max =Math.max(Math.max(pred[j][0], pred[j][1]),pred[j][2]);
							double min =Math.min(Math.min(pred[j][0], pred[j][1]),pred[j][2]);
							predAv[j][0]=Math.abs(max-min); //range
							predAv[j][1]=(double) j; //position
							predAv[j][2]=(pred[j][0]+pred[j][1]+pred[j][2])/3.0; //average prediction
							predAv[j][3] = (double)missingTruePos.get(j);
							
							//optimize
							pred[j]=null;
						}
						
						//pick the best unlabeled and add them to labeled
						Arrays.sort(predAv,new Comparator<Double[]>(){
							@Override
							public int compare(final Double[] entry1,final Double[] entry2){
								final Double range1 = entry1[0];
								final Double range2 = entry2[0];
								return range1.compareTo(range2);
							}
						});
						
						// Add best unlabeled to labeled and after remove them (there's a reason for 2 steps)
						int maxBestUnlabeled = Math.min(predAv.length, numOfBestUnlabeled); // unlabeled must be enough
						Integer[] pointerDeleteUnlabeled = new Integer[maxBestUnlabeled];
						//System.out.println("Added: "+maxBestUnlabeled);
						for(int k=0;k<maxBestUnlabeled;k++){
							//Use Missing Index
							double currentClassValue = df.get(new Double(predAv[k][3]).intValue()).value(l);
							double newClassValue = predAv[k][2];
							df.get(new Double(predAv[k][3]).intValue()).setValue(l, newClassValue);
							sumOfSquares += Math.pow(currentClassValue - newClassValue, 2);
							
							//System.out.println("aa"+Arrays.deepToString(predAv[k]));						
							unlabeled.instance(new Double(predAv[k][1]).intValue()).setClassValue(predAv[k][2]);
							labeled.add(unlabeled.instance(new Double(predAv[k][1]).intValue()));						
							pointerDeleteUnlabeled[k]=new Double(predAv[k][1]).intValue();
						}			
						// it's important to iterate from last to first, because when we remove
						// an instance, the rest shifts by one position.
						Arrays.sort(pointerDeleteUnlabeled);
						//System.out.println(Arrays.deepToString(pointerDeleteUnlabeled));
						for(int k=maxBestUnlabeled-1;k>=0;k--) {
							//Check code
							//They both are in the same index
							/*
							 * System.out.println("\n\n--Start----");
							 * System.out.println(unlabeled.get(pointerDeleteUnlabeled[k]));
							 * System.out.println(df.get(missingTruePos.get(pointerDeleteUnlabeled[k])));
							 * System.out.println("\n\n--End----");
							 */
							unlabeled.delete(pointerDeleteUnlabeled[k]);
							missingTruePos.remove((int)pointerDeleteUnlabeled[k]);
						}
						
						System.out.println("Labeled: "+labeled.size()+" Unalebled: "+unlabeled.size());
						
					}
					
					mRegs.buildClassifier(labeled);
					
					/* Check Code
					 * mRegs.isMultiple = false; System.out.println("Test:"+mRegs.isMultiple+" "
					 * +((ParallelMultipleRegressor)cls).isMultiple);
					 */
					
					System.out.println("Regression Selftrain OUTSIDE");
					
					
				}
				
				
				
				

				if (sumOfSquares < m_epsilon)
					isStable[l] = true;

				/*
				 * Step 6: Carry out steps 4-5 in turn for each l = 2..p.
				 */
			}

			boolean allStable = true;
			for (int j = 0; j < isStable.length; j++) {
				if (j != originalClassIndex) {
					if (!isStable[j]) {
						allStable = false;
						break;
					}
				}
			}
			if (allStable)
				break;
		}

		// temp header for imputation time
		m_Header = new Instances(data, 0);

		return new Instances(data, 0);
	}

	/**
	 * Performs the actual imputation.
	 *
	 * @param inst the instance to perform imputation on
	 * @return the updated instance
	 * @throws Exception if the imputation fails, eg if not initialized.
	 */
	@Override
	protected Instance doImpute(Instance inst) throws Exception {
		System.out.println("doImpute Has RUN!");
		Instance		result;
		int			i;

		result = (Instance) inst.copy();
		result.setDataset(m_Header);
		//System.out.println(inst);

		for (i = 0; i < result.numAttributes(); i++) {
			if (i == inst.classIndex())
				continue;
			if (result.isMissing(i)) {
				if (m_classifiers[i] != null) {
					m_Header.setClassIndex(i);
					if(m_classifiers[i] instanceof ParallelMultipleRegressor) { // if we have multiple classifiers its regression
						System.out.println("Predict with Regression");
						ParallelMultipleRegressor mRegs = (ParallelMultipleRegressor)m_classifiers[i];
						mRegs.setNumExecutionSlots(3);
						result.setValue(i, (mRegs.getClassifier(0).classifyInstance(result)
											+mRegs.getClassifier(1).classifyInstance(result)
											+mRegs.getClassifier(2).classifyInstance(result)
											)/3.0);
					}else {
						System.out.println("Predict with Classification");
						result.setValue(i, m_classifiers[i].classifyInstance(result));
					}
				}
			}
		}

		result.setDataset(m_OutputFormat);

		return result;
	}
}
