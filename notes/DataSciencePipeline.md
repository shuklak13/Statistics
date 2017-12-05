* Relational Databases
    * Why use SQL instead of NoSQL?
        * Enforcement of ACID (Atomicity, Consistency, Isolation, Durability) on database transactions
        * Easy-to-understand relational model
        * Powerful JOIN clause allows for operations over many combinations of large sets of data
    * Why use NoSQL instead of Relational?
        * Data with dynamic structure that cannot be easily captured in a tabular format
        * Better scaling w/ large quantities of data (b/c less overhead)
    * SQLite
        * a very fast, small, lightweight DB
        * SQLite is a C library that gets "embedded" inside that uses it
            * also comes with a CLI that can be used to modify the database file
        * a "file-based" DB instead of a "process-based" DB
            * when an app uses SQLite, data is sent to a file rather than to sockets
            * only one cross-platform file is used to store the entire database
        * useful for embedded apps - those that need portability w/o external dependencies, such as small mobile apps or games
            * also useful for testing without using a full DB
        * however, SQLite has no user management system
            * not useful for an application that requires concurrent or distributed access (all data stored in 1 file on 1 host, so only 1 write may occur at a time)
        * dynamically and weakly typed, unlike most RDBMS, which are staticlly and strongly typed
        * very popular - used in most browsers (Chrome, Firefox, Opera, Safari), some web frameworks (Django, Ruby on Rails), and some operating systems (Android, Win10)
    * MySQL
        * most popular DB - commonl in web apps
        * easy, feature-rich, scalable, and secure
    * MariaDB
        * a branch of MySQL created by MySQL's original developers after Oracle purches Sun (MySQL's original creator)
            * almost exactly identical, except rather than relying on Oracle's official support you are part of the open source community
        * pretty much better than MySQL unless you want Oracle support
    * PostegreSQL
        * considered by some to be the most powerful/advanced DB, with many features
        * highly extendible with stored procedures
        * provides superior data reliability to MySQL
        * however, it is slower than MySQL for read ops, so it's not the best tool if you just need a simplistic read-heavy datastore
        * it can be overkill if you have a simple set-up

* MongoDB
    * MongoDB vs ElasticSearch:
        * https://stackoverflow.com/questions/29538527/mongodb-elasticsearch-or-only-elasticsearch
        * MongoDB is better as a persistent data store. It can be used as a primary store.
        * Elasticsearch is an indexing service that makes it very easy to search and query data. Elasticsearch is better as an additional service for fast querying on top of a primary store.
            * Elasticsearch's querying is more powerful because it has a ranking feature allowing for sorted results. It can use domain-specific knowledge to create smarter queries than a simple keyword search.
    * MongoDB uses "documents" which are key-val stores similar to Elasticsearch
    * MongoDB units
        * Collections (like a table in RDBMs or an Index in Elasticsearch)
        * Document    (like a row in RDBMs)
        * Fields      (like a column in RDBMs)
    * Advantages over RDBMs
        * Schemaless (schema may defer from one doc to another, even within the same collection)
        * Easy to scale   (auto-sharding, replication)
        * JSON-style documents
        * Can index on any document
    * Components
        * mongod.exe - run this to start the MongoDB server
        * mongo.exe - run this to interact with the running MongoDB instance through the CLI
    * Secondary Indexing
        * MongoDB allows you to index on any field
        * Drawbacks: the more indices you have, the slower writes become (in exchange for faster reads)
            * Generally, the read boost is way bigger than the write penalty. It's good to have multiple indices, so longa s you make sure that all of them are actually useful


# Purpose of this document

* Explain what each of these technologies does
* Explain when/why you would you use these technologies
* Explain how these technologies fit together into the data science pipeline


# Database Terminology

## Databases

*  Why do we need a database?
    * They provide a layer of abstraction and insulation between the program and the data
    * They provide efficient and convenient querying and aggregating
    * They allow data sharing between multiple applications
    * They provide backup and recovery functionality
* When would you not want to use a DBMS?
    * If you have stringent real-time or storage requirements that may be hindered the database overhead
    * When your dataset is being used by a single user or application and won't need to be shared

## [CAP Theorem](https://en.wikipedia.org/wiki/CAP_theorem) (where to put this?)

* A distributed data system can only provide 2 of the following 3:
    * *Consistency*: the system will always return the same result for a given query, regardless of where the query was made
    * *Availability*: the system and its data will always be available for access, even if individual machines in the system fail
        * in reality, Availability is almost always required; this means the choice turns into Consistency vs Partition-Tolerance
    * *Partition-Tolerance*: even if the system fractures due to a loss in connection between nodes, it will remain useable

## [ACID](https://en.wikipedia.org/wiki/ACID)

* A database transaction (group of operations) should have the following properties:
    * *Atomicity*: each transaction is treated as an indivisible unit - if the transaction fails, all operations within it fail; if it succeeds, all of its operations succeed
    * *Consistency*: each transaction must take the database from one valid state to another; no rule defined for the database may be violated
    * *Isolation*: transactions can operate concurrently with the same result as if they operated sequentially
    * *Durability*: after a transaction finishes, it should be stored in non-volatile memory to prevent data loss

## Normalization

* used in relational databases
* 1st Normal Form   (all attributes depend on the key)
    * all attributes are atomic
    * to achieve this, split composite attributes into multiple atomic attributes
        * if an attribute is nested, create a new table from that attribute
* 2nd Normal Form   (all attributes depend on the entire key)
    * the table has no partial functional dependencies - that is, if the primary key is composed of multiple attributes, then all other attributes in the table are dependent on the unique combination of primary key's attributes
    * Example: (put this in later if I feel like it)
    * to achieve this, create a new table for every partial functional dependency, using the unique subset of the primary key as the new table's primary key
* 3rd Normal Form   (all attribtues depend on nothing but the key)
    * no attribute can be dependent on another attribute, unless that attribute is the key
    * to achieve this, create a new table for every dependency that is not based on the table's key
* Decomposition
    * the process of splitting a dataset's universal relation schema (the schema containing all of a datasets attributes) into tables using normalization rules


# Technologies

## SQL

* the structured query langauge - this is the basis for most database querying
* based on the SELECT-FROM-WHERE syntax

## Apache Hadoop

* Distributed processing platform + file system for large data sets using MapReduce
* Uses the Hadoop Distributed File System (HDFS), a distributed file system
* Properties
    * highly-scalable and redundant, ensuring high availability by detecting + handling failures
    * files are broken into blocks are replicated across multiple nodes
    * optimized for write-once, read-many applications
* Resource management and job scheduling is handled by [Yarn](https://hortonworks.com/apache/yarn/), the "central platform" for consistency, security, and data governance in Hadoop
* HDFS Architecture
    * Name Node AKA Master Node
        * stores metadata and the location of data
        * the interaction point for client applications
        * single point of failure
    * Data Node AKA Slave Node AKA Worker Node
        * stores data and performs operations
    * Job Tracker
        * gets data locations from Master Node and assigns tasks to the Task Tracker
    * Task Tracker
        * accepts tasks from the Job Tracker and assigns them to Data Nodes
        * gives status updates to the Job Tracker
* At what size should I begin using Hadoop?
    * Hadoop is *not* a cure-all. It's bound to the MapReduce paradigm, limiting the type of applications it can be used for to mostly batch processing, and its high overhead means that it's not really good for small computations.
    * [Generally, a scale of several Terrabytes is when Hadoop becomes the optimal tool. At smaller sizes, buying a computer with a larger RAM and/or disk is usually a smarter option](](https://www.chrisstucchio.com/blog/2013/hadoop_hatred.html))
* Some other projects built on top of Hadoop include...
    * Pig, a scripting interface on top of MapReduce that provides a more concise and user-friendly syntax; however, Pig scripts are typically slower than well-written MapReduce jobs. Features include...
        * explicit data flow (unlike sQL)
        * procedural programming language (unlike MapReduce)
        * lazy evaluation (like Spark)
        * nested data models
        * relational algebra operations
    * Hive, a SQL interface on top of MapReduce; however, [Hive is even slower than Pig, so don't use it for efficiency!](https://www.ibm.com/developerworks/library/ba-pigvhive/pighivebenchmarking.pdf)
    * Cassandra, a decentralized distributed NoSQL database built on HDFS
        * [Because Cassandra is decentralized, it has no single point of failure, allowing for greater Partition-Tolerance than HBase. Cassandra is more optimal for write operations. Both databases provide high availability.](http://bigdatanoob.blogspot.in/2012/11/hbase-vs-cassandra.html)
    * HBase, a centralized distributed NoSQL database built on HDFS modeled after [Google Bigtable](https://static.googleusercontent.com/media/research.google.com/en//archive/bigtable-osdi06.pdf) with a Java API
        * [Because HBase is centralized, there is a guarantee of consistency, unlike Cassandra. HBase is more optimal for read operations. Both databases provide high availability.](http://bigdatanoob.blogspot.in/2012/11/hbase-vs-cassandra.html)
    * Storm, a distributed stream processing framework
        * vs Spark?
            * Storm focuses on task-paralllel computations, unlike Spark which does data-parallel computations
            * Storm does batch real one-at-a-time stream processing; Spark does stream processing in mini-batches, which is worse than Storm for data ingested in incredibly low latencies but is better fr processing stable data

### MapReduce

* Broad Idea: split a large dataset of key-value across many computational units (map), then combine their results (reduce).
* Key Principles of MapReduce programming:
    * isolation - minimize communication between tasks in order to reduce latency
    * stateless - information is discarded after an operation is performed
    * data distribution and task parallelization are hidden from the user
    * batch-oriented - bandwidth matters more than latency
    * consistency is more important than availability - if a node goes down, the system will slow down
* Phases:
    * Map: apply a function that modifies all pairs and writes them to memory
        * the mapper nodes use a circular buffer in RAM to hold Intermediate Outputs (IOs) - once the buffer fills up, it spills to disk
    * Combine: an optional phase that performs a Reduce function in-disk
        * may run once, never, or multiple times - this is purely an optimization to reduce the amount of data that needs to be processed in the reduce phase
    * Shuffle
        * Partition: determines which Reducer gets which iO
        * Sort: determines in what order the Reducer gets an IO
            * by default, there is no order - specifying an order reduces the performance of MapReduce, so its recommended not to use it unless the order in which data is read actually matters
    * Reduce: pply a function that aggregates all pairs with the same key
* A "Mapper" or "Reducer" is a function written in the MapReduce framework -  a Map Task or a Reduce Task is a running instance of those functions executing on a worker node
* After every map() or reduce(), data is written back to disk. This is a slow process.
* The Execution Framework handles several things without user intervention, including...
    * Task Scheduling and Data Distribution (the user doesn't decide which node data goes to, or where different tasks run)
    * Synchronization/Shuffling (this may be programmed, but is not necessary)
    * Fault-Tolerance
* MapReduce has many of its own design patterns, including Local Aggregation, Pairs, Stripes, Order Inversion, and Secondary Sorting - 
* Downside: In order to perform iterative processes, many MapReduce jobs must be chained together. This repeated reading-and-writing-from-disk is extremely slow, which is why people use Spark.

## Apache Spark

* An in-memory cluster-computing tool that uses "Resilient Distributed Datasets" (RDDs)
* RDDs
    * Spark's primitives - they are consistent, immutable, partitioned, distributed, shared memory abstractions
    * built for in-memory parallel computations on clusters
    * 3 Types of RDD operations
        * Transformations: operations that simply modify data without returnin an output, like map, filter, or distinct
            * these are lazily-evaluated, meaning you can specify several transformation operations without having to wait for them to execute (until an action operation is given, at least)
        * Actions: operations that do return an output, like count, reduce, collect, or take
            * these are eagerly-evaluated - all the transformations on a particular RDD will be executed along with the given action
        * Persistence: operations that specify whether data should be stored in memory or disk, whether it should be serialized, and whether it should be replicated. Examples include persist and cache.
            * the ability to specify the storage level of a data structure is what part of what makes Spark so effective at iterative algorithm
    * RDDs' internals include...
        * its dependencies (the RDDs, or data it was built from)
        * the function to compute the RDD's data
        * the data partitioning
* [Alternatives to RDDs](https://databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html)
    * since the creation of RDDs, the Spark project has added two new structures for users to interact with data - DataFrames and Datasets.
    * RDDs
        * RDDs are immutable and distributed sets of data operated on in memory
        * the RDD API provides low-level operations
        * this makes it good for working with unstructured data, such as streams of text
        * the downside of RDDs are that they are JVM objects - this means they must be garbage-collected and Java-serialized
            * Serialization is when a JVM object is converted into a bytestream; Deserialization is when the bytestream is read and converted back into a JVM object
                * Serialization is used for transmitting data to memory, files or databases; obviously, for distributed systems, efficient serialization is essential
                * A part of memory that stores serialized objects is known as the *off-heap store*
                    * data in off-heap store (serialized data) is slower to access than on-heap data, but faster than data in disk store
    * DataFrames and Datasets
        * built on top of RDDs, DataFrames and Datasets provide a tabular view of data in a similar manner to the data frames of R and pandas
            * they use SparkSQL's "Catalyst" optimizer, making them more space and time efficient than RDDs (though at the sacrifice of not having access to lower-level operations)
            * [this page](https://databricks.com/blog/2015/02/17/introducing-dataframes-in-spark-for-large-scale-data-science.html) has some examples on how to use DataFrame operations
        * the rows of DataFrames are untyped, while the rows of Datasets are strongly-typed
            * note that Python and R can only use DataFrames, while Java can only use Datasets; Scala can use both
    * RDDs, DataFrames, and Datasets can be converted to each other easily
* 2 Types of Shared Variables - in order to allow data to be used by all nodes with minimal performance penalties, we impose some usage limitations
    * Broadcast Variables: read-only variable, cached on each node
    * Accumulators: only additions are allowed
* Spark is more efficient than MapReduce for data sharing
    * MapReduce relies on stable storage (disk) to share data between processes
    * RDDs use memory, so data sharing between processes is much more efficient
* Differences from MapReduce
    * Runs in-memory, making repeated accesses faster. This makes Spark the superior choice over Hadoop for iterative processes.
    * However, this also means Spark is much more memory-intensive than Hadoop. When operating over data that is too large to fit in a single machine's memory, MapReduce is a better choice than Spark.
    * Spark's API is genearlly considered friendlier than Hadoop MapReduce's, with easier syntax and an interface for Python and Scala. Pig makes MapReduce more accessible, but it's also much slower than MapReduce, so it isn't as common.
    * Can run standalone, or on top of YARN.
* Spark also has several components, including...
    * Spark Streaming: a real-time data ingestion tool that can hook up with 
    * SparkSQL: a SQL interface for Spark
    * SparkML: a machine learning library that takes advantage of Spark's RDD structure
    * GraphX: a Spark API for graph-based computations

## [Apache Zookeeper](https://en.wikipedia.org/wiki/Apache_ZooKeeper)

* a fast and scalable distributed configuration and synchronization service for large distributed systems
* stores the status of running processes in local log files - essentially a key-value store under the hood
* initially created as a sub-project of Hadoop, it is now used extensively in other applications

## Elasticsearch/Solr/Lucene

* [Apache Lucene](https://lucene.apache.org/core/) is a highly performant open source text search engine, capable of searching large amounts of text with relatively small RAM requirements
* Elasticsearch and Apache Solr are two open source applications built on top of the Apache Lucene open-source search-based document-store database
* Why would I use these?
    * You have a query-heavy application and want a data store with very robust querying capabilities and where you can define your own schema
* Elasticsearch vs Solr
    * Apache Solr is the older product, created in 2004 and open-sourced in 2006. Elasticsearch was released in 2010.
        * This means that Solr has better documentation, a more established community, and more configuration options, but Elasticsearch has more features appealing to modern developers, particularly in cloud/distributed environments
    * Elasticsearch is built for cloud, with distributed storage, multitenant support, and a built-in Zookeeper-like component called "Zen"
    * Elasticsearch has stronger querying, filtering, grouping, and aggregating capabilities
    * Interface
        * [Elasticsearch's interfacing DSL is a RESTful API where queries are formatted as JSON objects, making it more intuitive for modern developers while Solr's language is much more terse, similar to Perl](http://opensourceconnections.com/blog/2016/01/22/solr-vs-elasticsearch-relevance-part-two/)
        * Elasticsearch has a [variety of officially-supported clients for different languages](https://www.elastic.co/guide/en/elasticsearch/client/index.html), while Solr is limited to Java
    * Solr has access control functionality, but Elasticsearch does not
* Elasticsearch Terminology
    * Index: like a database
    * Type: like a table in a database - it has a list of Fields, defined by the Type's Mapping
    * Document: like a row - this is a single JSON object of a specified Type
    * Field: like a cell - these are comma-separated key:value pairs that belong to a Document
    * Mapping: similar to a RDBMS's schema definition
    * Shard: this is a single instance of Lucene - it is managed automatically by Elasticsearch, and is used for replication and fault-tolerance
* Percolator
    * One cool feature of ES is the "percolator", an inverse search where queries are stored and indexed (as opposed to documents), and are retrieved by inputting matching documents
        * Basically, you provide a document, and ES returns queries that the document would match to.
        * This can be useful for discovering queries if you know the type of document you're looking for but not sure what is the best query for it.

## Kafka/RabbitMQ

* These are messaging systems, meaning that they are used to transfer data between processes
* Kafka:
    * pub-sub (AKA broadcast)
    * dumb broker, smart consumers
        * the broker is a simple queue that consumers are programmed to read from
        * because of this, Kafka can support many consumers with very little overhead
    * superior choice for data streaming
    * built for durability (can access message history) and scalability (works well for large number of consumers)
    * relies on Zookeeper
* RabbitMQ:
    * general purpose message broker w/ several styles including point-to-point, request/repply, and pub-sub
    * smart broker, dumb consumers
        * the consumers simply pull from a queue, which the broker intelligently manages
    * superior choice if your application needs access to a variety of different messaging paradigms

# Data Infrastructure

* Small data: 
    * HOW BIG IS SMALL DATA????
* Regular batched (not real-time) ETL operations and big data analytics: Hadoop
* Real-time big data analytics: Spark

* Elasticsearch vs Hadoop
    * Elasticsearch is great for search integration in web apps
    * Analytics Backend?
        * Elasticsearch is gaining momentum because...
            * it's very easy to set up in small instances, compared to the bulkiness of Hadoop
            * its query language is way easier than MapReduce programming
        * [Elasticsearch has a common "split brain" problem where a cluster that gets disconnected will have multiple master nodes who might disagree with each other, causing a failure in consistency](https://blog.trifork.com/2013/10/24/how-to-avoid-the-split-brain-problem-in-elasticsearch/)
            * Because of this, if the integrity of data is essential, it should be stored in a fault-taulerant database (like Hadoop or MongoDB) and periodically replicated to Elasticsearch for querying and analytics
        * Elasticsearch is NOT good as an analytics backend - although it has fast query times, it has slow uploads, and lacks the robustness of MapReduce and Hadoop's many libraries