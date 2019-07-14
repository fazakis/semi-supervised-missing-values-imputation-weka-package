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

/*
 *    MSSRegression.java
 *    Copyright (C) 2017 University of Patras, Greece
 *
 */

package weka.classifiers.meta;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.ParallelMultipleClassifiersCombiner;
import weka.classifiers.trees.M5P;
import weka.core.Capabilities;
import weka.core.CommandlineRunnable;
import weka.core.Environment;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.Vector;

/**
 * <!-- globalinfo-start --> Class implementing multi-scheme regression.<br/>
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 * -B &lt;classifier specification&gt;
 *  Full class name of base classifier to include, followed
 *  by scheme options. May be specified multiple times.
 *  (default: -B "weka.classifiers.trees.M5P" -B "weka.classifiers.rules.M5P"
 *  -B "weka.classifiers.functions.SMOreg" -B "weka.classifiers.trees.RandomForest")
 * </pre>
 * 
 * <pre>
 * -do-not-check-capabilities
 *  If set, classifier capabilities are not checked before classifier is built
 *  (use with caution).
 * </pre>
 * 
 *
 * <pre>
 * -do-not-print
 *  If set, no individual models are printed in the output
 * </pre> 
 * 
 * 
 * <!-- options-end -->
 * 
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 * 
 * @author Nikos Fazakis (fazakis@ece.upatras.gr)
 * @version $Revision: 10000 $
 */
public class ParallelMultipleRegressor extends ParallelMultipleClassifiersCombiner implements
  TechnicalInformationHandler {

  /** For serialization */
  private static final long serialVersionUID = -637891196294399624L;
    
  /** Environment variables */
  protected transient Environment m_env = Environment.getSystemWide();

  /** Structure of the training data */
  protected Instances m_structure;

  /** Print the individual models in the output */
  protected boolean m_dontPrintModels;
  
  
  /**
   * Constructor.
   */
  public ParallelMultipleRegressor() {
	  Classifier[] defaultClassifiers = {
			    new weka.classifiers.trees.M5P(),
			    new weka.classifiers.functions.SMOreg(),
			    new weka.classifiers.trees.RandomForest()
			  };
	  setClassifiers(defaultClassifiers);
  }
  
  public ParallelMultipleRegressor(Classifier[] cls) {
	  Classifier[] defaultClassifiers = cls;
	  setClassifiers(defaultClassifiers);
  }

  /**
   * Returns a string describing MSSRegression
   * 
   * @return a description suitable for displaying in the explorer/experimenter
   *         gui
   */
  public String globalInfo() {
    return "Class implementing multi-scheme semi-supervised regression."
      + "\n\n"
      + "For more information see:\n\n" + getTechnicalInformation().toString();
  }

  /**
   * Returns an enumeration describing the available options.
   * 
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {

    Vector<Option> result = new Vector<Option>();
    
    /** These options are inherited**/
    result.addElement(new Option(
            "\tNumber of execution slots.\n"
            + "\t(default 1 - i.e. no parallelism)",
            "num-slots", 1, "-num-slots <num>"));
    
    result.addElement(new Option(
            "\tFull class name of classifier to include, followed\n"
            + "\tby scheme options. May be specified multiple times.\n"
            + "\t(default: -B \"weka.classifiers.rules.M5P\""
            +" -B \"weka.classifiers.functions.SMOreg\" -B \"weka.classifiers.trees.RandomForest\")",
            "B", 1, "-B <classifier specification>"));    
    
    final class Temp extends AbstractClassifier{
    	@Override
		public void buildClassifier(Instances data) throws Exception {}};
    result.addAll(Collections.list((new Temp()).listOptions()));
    /** End inherited options **/
    
    result.addElement(new Option(
    	      "\tSuppress the printing of the individual models in the output",
    	      "do-not-print", 1, "-do-not-print"));

    /** It gets too crowded with this super**/
    //result.addAll(Collections.list(super.listOptions()));

    return result.elements();
  }

  /**
   * Gets the current settings of MSSRegression.
   * 
   * @return an array of strings suitable for passing to setOptions()
   */
  @Override
  public String[] getOptions() {
    int i;
    Vector<String> result = new Vector<String>();
    String[] options;

    options = super.getOptions();
    for (i = 0; i < options.length; i++) {
      result.add(options[i]);
    }  

    if (m_dontPrintModels) {
      result.add("-do-not-print");
    }

    return result.toArray(new String[result.size()]);
  }

  /**
   * Parses a given list of options.
   * <p/>
   * 
   * <!-- options-start --> Valid options are:
   * <p/>
   * 
   * <pre>
   * -B &lt;classifier specification&gt;
   *  Full class name of classifier to include, followed
   *  by scheme options. May be specified multiple times.
   *  (default: -B "weka.classifiers.trees.M5P" -B "weka.classifiers.rules.M5P"
   *  -B "weka.classifiers.functions.SMOreg" -B "weka.classifiers.trees.RandomForest")
   * </pre>
   * 
   * <pre>
   * -do-not-check-capabilities
   *  If set, classifier capabilities are not checked before classifier is built
   *  (use with caution).
   * </pre>
   *
   * <pre>
   * -do-not-print
   *  If set, no individual models are printed in the output
   * </pre>
   * 
   * <!-- options-end -->
   * 
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {


    setDoNotPrintModels(Utils.getFlag("do-not-print", options));

    super.setOptions(options);
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing detailed
   * information about the technical background of this class, e.g., paper
   * reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  @Override
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation result;

    result = new TechnicalInformation(Type.UNPUBLISHED);
    result.setValue(Field.AUTHOR, "Nikos Fazakis");
    

    return result;
  }

  /**
   * Returns capabilities according to the selected classifiers.
   * 
   * @return the capabilities of this classifier
   */
  @Override
  public Capabilities getCapabilities() {
	Capabilities result = super.getCapabilities();
    result.disableDependency(Capability.NOMINAL_CLASS);
    result.disable(Capability.NOMINAL_CLASS);

    return result;
  }

  /**
   * Buildclassifier trains the classifiers iteratively and builds the final classifier.
   * 
   * @param data the training data to be used
   * @throws Exception if the classifier could not be built successfully
   */
  @Override
  public void buildClassifier(Instances data) throws Exception {
	  
	//TODO: support multiple base classifiers  
	if(m_Classifiers.length!=3)
		throw new Exception("Number of base classifiers should be 3 explicitly.");
	
		super.buildClassifier(data);
		try{
			super.buildClassifiers(data);
		}catch(Exception e){
			if(e.getMessage().toLowerCase().contains("all class values are the same"))				
				throw new Exception(e.getMessage()+"\nTry increasing labeled ratio!");
			else
				throw e;
		}		
		//getFinalClassifier().buildClassifier(data);	
  }
  
  

  /**
   * Classifies the given test instance.
   * 
   * @param instance the instance to be classified
   * @return the predicted most likely class for the instance or
   *         Utils.missingValue() if no prediction is made
   * @throws Exception if an error occurred during the prediction
   */
  @Override
  public double classifyInstance(Instance instance) throws Exception {
	  
	 return 0;
	 //return getFinalClassifier().classifyInstance(instance);
	
  }

  /**
   * Returns the tip text for this property
   * 
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String doNotPrintModelsTipText() {
    return "Do not print the individual trees in the output";
  }

  /**
   * Set whether to print the individual ensemble models in the output
   * 
   * @param print true if the individual models are to be printed
   */
  public void setDoNotPrintModels(boolean print) {
    m_dontPrintModels = print;
  }

  /**
   * Get whether to print the individual ensemble models in the output
   * 
   * @return true if the individual models are to be printed
   */
  public boolean getDoNotPrintModels() {
    return m_dontPrintModels;
  }

  /**
   * Output a representation of this classifier
   * 
   * @return a string representation of the classifier
   */
  @Override
  public String toString() {

    if (m_Classifiers == null) {
      return "ParallelMultipleRegressor: No model built yet.";
    }

    String result = "Parallel Multiple Regressor combines";
    result += " the prediction range of these base learners:\n";
    for (int i = 0; i < m_Classifiers.length; i++) {
      result += '\t' + getClassifierSpec(i) + '\n';
    }

    

    StringBuilder resultBuilder = null;
    if (!m_dontPrintModels) {      
      resultBuilder = new StringBuilder();
      resultBuilder.append(result).append("\nAll the models:\n\n");
      for (Classifier c : m_Classifiers) {
        resultBuilder.append(c).append("\n");
      }
      //resultBuilder.append(result).append("\nNot yet supported\n\n");

    }

    return resultBuilder == null ? result : resultBuilder.toString();
  }

  /**
   * Returns the revision string.
   * 
   * @return the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 10000 $");
  }

  /**
   * Main method for testing this class.
   * 
   * @param argv should contain the following arguments: -t training file [-T
   *          test file] [-c class index]
   */
  public static void main(String[] argv) {
    runClassifier(new ParallelMultipleRegressor(), argv);
  }
  
  
  /** ParallelMultipleRegressor specific options **/



}
