> This folder is correspoing to the "cat[1~n]_[CattaryKeyName].html" in “_include” folder
> The two partitions consist of a new category.
> A typical file in this(i,e, "cattegory" folder) should like this:
~~~
---
layout: default
---
{%include cat2_life.html%}   //the file in “_include” folder
~~~

Regarding the file in “_include” folder,pls refer to "catX_example.html".
DO NOT FORGET to modify the "header.html" in “_include” folder to link the final static category path, which is "{{ site.baseurl }}/catagory/[corresspongdingNameOfFileInThisFolder]/index.html"
