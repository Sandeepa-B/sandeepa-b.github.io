---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: false
---

<!-- {% if author.googlescholar %} -->
  
<!-- {% endif %} -->

{% include base_path %}
You can also find my papers on <u><a href="https://scholar.google.com/citations?user=oCqKAnsAAAAJ&hl=en">Google Scholar</a>.</u>
-----------
{% capture written_year %}'None'{% endcapture %}
{% for post in site.publications reversed %}
  {% capture year %}{{ post.date | date: '%Y' }}{% endcapture %}
  {% if year != written_year %}
    <b>{{ year }}</b>
    {% capture written_year %}{{ year }}{% endcapture %}
  {% endif %}
  {% include archive-single.html %}
{% endfor %}


