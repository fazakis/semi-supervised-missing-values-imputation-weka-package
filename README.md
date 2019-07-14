Semi-Supervised Missing Values Imputation
=========================================

This weka package includes the Iterative Robust Semi-Supervised based Imputation (IRSSI) for Missing Data algorithm. The package is based on the repository https://github.com/fracpete/missing-values-imputation-weka-package .


Missing Values Imputation
-------------------------

Weka package for missing values imputation (and injection) using various techniques.

The following two filters are available:

* `weka.filters.unsupervised.attribute.MissingValuesImputation` - for imputing missing values
* `weka.filters.unsupervised.attribute.MissingValuesInjection` - for injecting missing values


Imputation
----------

The imputation techniques listed below are available through the
`weka.filters.unsupervised.attribute.MissingValuesImputation` filter:

* `NullImputation` - dummy
* `MeansAndModes` - like WEKA's `ReplaceMissingValues` filter
* `MultiImputation` - applies the specified imputation algorithms sequentially
* `SimpleNearestNeighbor` - uses nearest neighbor approach to determine most
   common label or average (date/numeric)
* `UserSuppliedValues` - simply replaces missing values with user-supplied ones
* `IRMI` - [M. Templ et al (2011): Iterative stepwise regression imputation
   using standard and robust methods](http://www.statistik.tuwien.ac.at/public/filz/papers/CSDA11TKF.pdf)
   (contributed by [Chris Beckham](https://github.com/christopher-beckham/weka-fimi))
* `IRSSI` - [Submitted IEEE ACCESS (2019): Iterative Robust Semi-Supervised based
   Imputation (IRSSI) for Missing Data]


Injection
---------

The injection techniques listed below are available through the
`weka.filters.unsupervised.attribute.MissingValuesInjection` filter:

* `NullInjection` - dummy
* `MultiInjection` - applies the specified injection algorithms sequentially
* `AllWithinRange` - set all specified attributes to missing
* `ClassOnly` - only sets the class values to missing
* `RandomPercentage` - sets random percentage of values in selected attribute range to missing


Releases
--------

Click on one of the following links to download the corresponding Weka package:

* [2019.7.14](https://github.com/fazakis/semi-supervised-missing-values-imputation-weka-package/releases/download/v2019.7.14/semi-supervised-missing-values-imputation-2019.7.14.zip)


How to use packages
-------------------

For more information on how to install the package, see:

https://waikato.github.io/weka-wiki/packages/manager/


Maven
-----

Add the following dependency in your `pom.xml` to include the package:

```xml
    <dependency>
      <groupId>com.github.fazakis</groupId>
      <artifactId>semi-supervised-missing-values-imputation-weka-package</artifactId>
      <version>2019.7.14</version>
      <type>jar</type>
      <exclusions>
        <exclusion>
          <groupId>nz.ac.waikato.cms.weka</groupId>
          <artifactId>weka-dev</artifactId>
        </exclusion>
      </exclusions>
    </dependency>
```

Please note, when using Maven you may have to register the imputation/injection
class hierarchies with Weka's GenericObjectEditor if you want to use them in the
GUI as well. See the following files:

* `GenericPropertiesCreator.props`
* `GUIEditors.props`
