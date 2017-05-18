# Cheatsheet
http://zevross.com/blog/2014/08/04/beautiful-plotting-in-r-a-ggplot2-cheatsheet-3

# Layers
* graphs are created from transparent "layers" which contain geometric shapes
* the x and y variables must be mentioned in the top layer, ggplot, only
* Example: `myGraph <- ggplot(myData, aes(x variable, y variable)) + geom_bar() + geom_point()`

# Aesthetics
* Each layer may have aesthetics
* Aesthetics added to the top layer (ggplot) modify the aesthetics of all the layers of a graph at the same time!
	* This is used to specify the specify the variable at each axis
		* `ggplot(myData, aes(x variable, y variable))`
* Aesthetics can be added with aes() to make them dynamic rather than static
	* Dynamic Example: `geom_point(aes(colour = gender))`
	* Static Example: `geom_point(colour = "Blue"))`

# Stats
* Components of layers with modifiable properties

# Themes
* Replaces the old opts() function
* Click [here](https://05154779709936255585.googlegroups.com/attach/5aa16afece3d5bc6/theme0.html?gda=9XgBzEYAAABCncUW0npTUN_veVgl3inYi0oNsf4Sjxsz8g3AimkTHy2Q5nwgitdzQrQMmMK7aytx40jamwa1UURqDcgHarKEE-Ea7GxYMt0t6nY0uV5FIQ&view=1&part=4) for more info

# Other Visualization Tools
* position - a parameter that makes it easier to deal with overlapping elements
	* Ex: `ggplot(myData) + geom_point(aes(colour = gender), position = "jitter")`
* facet_grid() and facet_wrap() - can be used to display a grid of multiple plots in one visualization
