---
layout: page
title: GDELT+Finance  
tagline: 
description: a COS 513 Project by Ghassen Jerfel, Zi Xiang Pan and Tom Wu 
---    
## Past Posts

<ul class="posts">
  {% for post in site.posts %}
    <li><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ BASE_PATH }}{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>



