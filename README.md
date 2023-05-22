# **Sagemaker-ML-Insights**

The project deploys a [Amazon SageMaker Studio](https://aws.amazon.com/sagemaker/studio/) domain and foundational infrastructure to query and load [Amazon Security Lake](https://aws.amazon.com/security-lake/). Once deployed, you can use SageMaker notebooks to run machine learning analytics against your security data to identify trends, anomalies and patterns.

By running machine learning analytics specific to your AWS environment, you will be able to quickly deploy and utilize SageMaker's capabilities to explore and derive ML powered insights from Security Lake data. This will enable you to idenfity different areas of interest to focus on and increase your overall security posture. The solution has a base set of notebooks that are meant to serve as a starting point and looks at [AWS Security Hub](https://docs.aws.amazon.com/securityhub/latest/userguide/what-is-securityhub.html) findings but can be expanded to incorporate other Security Lake native or custom data sources.
<br>

## **Prerequisites**

1. [Enable Amazon Security Lake](https://docs.aws.amazon.com/security-lake/latest/userguide/getting-started.html). For multiple AWS accounts, it is recommended to manage [Security Lake for AWS Organizations](https://docs.aws.amazon.com/security-lake/latest/userguide/multi-account-management.html) To help automate and streamline the management of multiple accounts, we strongly recommend that you integrate Security Lake with AWS Organizations.
2. As part of this solution, we are focusing on AWS Security Hub so you will need to enable [AWS Security Hub](https://docs.aws.amazon.com/securityhub/latest/userguide/securityhub-settingup.html) and configure [AWS Security Hub](https://docs.aws.amazon.com/security-lake/latest/userguide/securityhub-integration.html) as a data source to Security Lake.
3. [Subcriber Query Access](https://docs.aws.amazon.com/security-lake/latest/userguide/subscriber-query-access.html): Subscribers with query access can query data that Security Lake collects. These subscribers directly query AWS Lake Formation tables in your S3 bucket with services like Amazon Athena.
4. Resource Linking: Create a Lake Formation database in Machine Learning (ML) insights Subcriber account using resource linking
    - Go to Lake Formation in the ML Insights Subscriber AWS account
    - Create a new database using resource linking
    - Enter Resource Link name
    - Enter Shared database name and shared database Owner ID and click create
<br><br>

## **Solution Architecture**
![Solution Architecture](/sagemaker_ml_insights_architecture.png)

1. Security Lake is setup in a separate AWS account with the appropriate sources (i.e. VPC Flow Logs, Security Hub, CloudTrail, Route53) configured.
2. Create subscriber query access from source Security Lake AWS account to ML Insights Subscriber AWS account.
3. Resource share request accepted in the Subscriber AWS account where this solution is deployed.
4. Create a database in Lake Formation in the Subscriber AWS account and grant access for the Athena tables in the Security Lake AWS account.
5. VPC is provisioned for SageMaker with an IGW, NAT GW, and VPC endpoints for all AWS services within the solution. IGW/NAT is required to install external open-source packages.
6. SageMaker Studio Domain is created in VPCOnly mode with a single SageMaker user-profile that is tied to a dedicated IAM role. As part of the SageMaker deployment, an EFS also gets provisioned for the SageMaker Domain.
7. A dedicated IAM role is created to restrict access to create/access SageMaker Domain's presigned URL from a specific CIDR for accessing the SageMaker notebook.
8. CodeCommit repository containing python notebooks utilized for the AI/ML workflow by the SageMaker user-profile.
9. Athena workgroup is created for Security Lake queries with a S3 bucket for output location (Access logging configured for the output bucket).
<br><br>

## **Deploy Sagemaker Studio using CDK**

**Build**

To build this app, you need to be in the cdk project root folder [`source`](/source/). Then run the following:

    $ npm install -g aws-cdk
    <installs AWS CDK>

    $ npm install
    <installs appropriate packages>

    $ npm run build
    <build TypeScript files>

**Deploy**

    $ cdk bootstrap aws://<INSERT_AWS_ACCOUNT>/<INSERT_REGION>
    <build S3 bucket to store files to perform deployment>

    $ cdk deploy SageMakerDomainStack
    <deploys the cdk project into the authenticated AWS account>

As part of the CDK deployment, there is an Output value for the CodeCommit repo URL (sagemakernotebookmlinsightsrepositoryURL). You will need this value later on to get the python notebooks into your SageMaker app.

## **Post Deployment Steps**

**Access to Security Lake**

Now that you have deployed the SageMaker solution, you will need to grant SageMaker's user-profile in your AWS account access to query Security Lake from the AWS account it was enabled in. We will use the "Grant" permisson to allow the Sagemaker user profile ARN to access Security Lake Database in Lake Formation within the ML Insights Subscriber AWS account.

**Grant permisson to Security Lake Database**
1. Copy ARN “arn:aws:iam::********************:role/sagemaker-user-profile-for-security-hub” 
2. Go to Lake Formation in console
3. Click on amazon_security_lake_glue_db_us_east_1 database
4. From Actions Dropdown choose Grant
5. In grant Data Permissions select SAML Users and Groups
6. Paste the SageMaker Domain user profile ARN
7. In Database Permissions choose Describe and click on Grant 
<br><br> 

**Grant permisson to Security Lake - Security Hub Table**
1. Copy ARN “arn:aws:iam::********************:role/sagemaker-user-profile-for-security-hub” 
2. Go to Lake Formation in console
3. Click on amazon_security_lake_glue_db_us_east_1 database and the click on view tables button
4. Click on amazon_security_lake_table_us_east_1_sh_findings table
5. From Actions Dropdown choose Grant
6. In grant Data Permissions select SAML Users and Groups
7. Paste the SageMaker Domain user profile ARN
8. In Table Permissions choose Describe and select then click on Grant
<br>

**CodeCommit**
- Note: The Output (sagemakernotebookmlinsightsrepositoryURL) from the CDK deployment will have the CodeCommit repo URL.

##### Option 1: 
1. Open your SageMaker Studio app 
2. In Studio, in the left sidebar, choose the Git icon (identified by a diamond with two branches), then choose Clone a Repository.
3. For the URI, enter the HTTPS URL (Output value for SageMakerDomainStack.sagemakernotebookmlinsightsrepositoryURL) of the CodeCommit repository, then choose Clone.
4. In the left sidebar, choose the file browser icon. You will see a folder with the notebook repository

##### Option 2:
1. Open your SageMaker Studio app 
2. In the top navigation bar, choose File >> New >> Terminal
3. Type in the following command: 

    `$ git clone <'Output value for SageMakerDomainStack.sagemakernotebookmlinsightsrepositoryURL'>`
    <clones notebook repository>

<br>

## **Generating ML Insights using Sagemaker Studio**
Now that you have completed the post deployment steps. You are ready to start generating ML insights. The python notebooks which are deployed as part of the solution provide a starting point for how you can conduct AI/ML analysis using data within Security Lake. These can be expanded to any native or custom data sources configured on Security Lake.
<br>

**Environment Setup:**
![0.0-tsat-environ-setup](source/notebooks/tsat/0.0-tsat-environ-setup.ipynb)
- Installs all the required libraries and dependencies that are needed for notebooks to run Trend Detector, Outlier Detection and changepoint Detection.

**Load Data:**
![0.1-load-data](source/notebooks/tsat/0.1-load-data.ipynb)
- Establish the Athena connection to query data in Security Lake and create a time series dataset.
- Note : This notebook queries Security Hub Findings data from Security Lake and can be extended to any other Security Lake data source. To do so, change the TABLE parameter from security hub findings to any desired data source within Security Lake.
    
**Trend Detector:**
![1.1-trend-detector](source/notebooks/tsat/1.1-trend-detector.ipynb)
- Trend represents a directional change in the level of a time series. This directional change can be either upward (increase in level) or downward (decrease in level)
- Slopes are used identify the relationship between x(time) & y(counts). The trend is up when the slope is positive and is down when the slope is negative.
- Trend detection helps detect a change, while ignoring the noise from natural variability. Each environment is different and trends help us identify where to look more closely to determine why that trend is positive or negative.

**Outlier Detection:**
![1.2-outlier-detection](source/notebooks/tsat/1.2-outlier-detection.ipynb)
- We do a seasonal decomposition of the input time series, with additive or multiplicative decomposition as specified (default is additive).
- We generate a residual time series by either removing only trend or both trend and seasonality if the seasonality is strong.
- We detect points in the residual which are outside 3 times the inter quartile range. This multiplier can be tuned using the iqr_mult parameter in OutlierDetector.
- The intent is to discover useful, abnormal, and irregular patterns within in data sets, allowing you to pinpoint areas of interest.

**Change point Detection:**
![1.3-changepoint-detector](source/notebooks/tsat/1.3-changepoint-detector.ipynb)
- Change point detection is a method to detect sudden changes in a time series that persist over time, such as a change in the mean value.
- To detect a base line to identify when several changes might have occurred from that point.

## **Security**

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## **License**

This library is licensed under the MIT-0 License. See the LICENSE file.
