---
layout: archive
title: "技术笔记"
permalink: /foundations/
author_profile: false
---

<ul class="index-list project-index-list foundation-index-list">
{% for foundation in site.data.foundations %}
  <li>
    {% if foundation.teaser %}
      <a class="list-thumb list-thumb--project" href="{{ foundation.url | relative_url }}" aria-label="{{ foundation.name }}">
        <img src="{{ foundation.teaser | relative_url }}" alt="{{ foundation.teaser_alt | default: foundation.name }}">
      </a>
    {% endif %}
    <div class="project-index-list__body">
      <a class="project-index-list__title" href="{{ foundation.url | relative_url }}">{{ foundation.name }}</a>
      <span>{{ foundation.display_date }} · {{ foundation.status }} · {{ foundation.highlight }}</span>
      <p>{{ foundation.summary }}</p>
      {% if foundation.tags and foundation.tags.size > 0 %}
        <div class="foundation-index-list__tags" aria-label="{{ foundation.name }} tags">
          {% for tag in foundation.tags %}
            <span>{{ tag }}</span>
          {% endfor %}
        </div>
      {% endif %}
    </div>
  </li>
{% endfor %}
</ul>
