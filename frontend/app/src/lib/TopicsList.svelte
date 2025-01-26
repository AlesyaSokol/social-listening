<script>
  let {topics} = $props();
  let activeIndex = $state(null);

  function toggle(index) {
    activeIndex = activeIndex === index ? null : index;
  }

function convertDateToRussianText(dateString) {
  const months = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря"
  ];

  // Parse the input date (assuming the format is DD-MM-YYYY)
  const [day, month, year] = dateString.split('-');

  return `${day} ${months[parseInt(month) - 1]} ${year}`;
}
</script>

<ul>
  {#each topics as topic}
    <li>
      <div class="topic-name" on:click={() => toggle(topic.topic_gpt)}>
        <div class="name-container">
          <span class="name {activeIndex === topic.topic_gpt ? 'active' : ''}">{topic.topic_gpt}</span>
          <div class="count">{convertDateToRussianText(topic.date)}</div>
        </div>
        <span style="text-align: right">
        {#each {length: Math.ceil(topic.n_posts / 20)} as _, i}
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512"><!--!Font Awesome Free 6.7.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2025 Fonticons, Inc.--><path d="M159.3 5.4c7.8-7.3 19.9-7.2 27.7 .1c27.6 25.9 53.5 53.8 77.7 84c11-14.4 23.5-30.1 37-42.9c7.9-7.4 20.1-7.4 28 .1c34.6 33 63.9 76.6 84.5 118c20.3 40.8 33.8 82.5 33.8 111.9C448 404.2 348.2 512 224 512C98.4 512 0 404.1 0 276.5c0-38.4 17.8-85.3 45.4-131.7C73.3 97.7 112.7 48.6 159.3 5.4zM225.7 416c25.3 0 47.7-7 68.8-21c42.1-29.4 53.4-88.2 28.1-134.4c-4.5-9-16-9.6-22.5-2l-25.2 29.3c-6.6 7.6-18.5 7.4-24.7-.5c-16.5-21-46-58.5-62.8-79.8c-6.3-8-18.3-8.1-24.7-.1c-33.8 42.5-50.8 69.3-50.8 99.4C112 375.4 162.6 416 225.7 416z"/></svg>
        {/each}
        <div class="count">{topic.n_posts} публ.</div>
        </span>
      </div>
      <div class="content {activeIndex === topic.topic_gpt ? 'open' : ''}">
        <div>
          {#each {length: 3} as _, i}
          <div class="post">{topic.post_texts[i].slice(0,150)}...<br><a href={topic.links[i]}>читать целиком</a></div>
          {/each}
        </div>
        <div class="regions">
          Обсуждают в регионах:
          {#each new Set(topic.regions) as region}
          <span>{region}; </span>
          {/each}
        </div>
      </div>
    </li>
  {/each}
  </ul>

<style>
  ul {
    list-style: none;
    text-align: left;
    padding: 0;
    margin: 0;
    min-width: 375px;
    max-width: 500px;
  }
  ul li {
    border-bottom: 1px solid #dddee1;
    animation: all .5s ease;
  }
  ul li:first-child {
    border-radius: 10px 10px 0 0;
  }
  ul li:last-child {
    border-bottom: none;
    border-radius: 0 0 10px 10px;
  }
  ul li .name-container {
    max-width: 75%;
  }
  ul li .count {
    font-size: 13px;
    color: #888;
  }
  .topic-name {
    display: flex; 
    justify-content: space-between;
    padding: 8px 16px;
  }
  .topic-name:hover {
    cursor: pointer;
    background-color: #dddee1;
  }
  svg {
    height: 13px;
    fill: #7fbbf3;
  }
  .name {
    font-size: 15px;
  }
  .name.active {
    color: #1576d4;
  }
  .content {
    display: none;
    padding: 16px;
  }
  .content.open {
    display: block;
  }
  .regions {
    color: #888;
    font-size: 11px;
  }
  .post {
    margin-bottom: 8px;
    font-family: monospace;
  }
</style>
