<script>
  import TopicsList from './lib/TopicsList.svelte';
  import Map from './lib/Map.svelte';

  let topics = [
    {id: 1, name: "test name1 test name1 test name1 test name1", hot: 2, posts_count: '390'},
    {id: 2, name: "test_name2", hot: 1, posts_count: '200'},
    {id: 3, name: "test_name3", hot: 3, posts_count: '300'}
  ];
</script>

<main>
  <div class="container">
    <Map topics={topics} />
    <div class="card-container">
      <div class="card">
        <TopicsList topics={topics} />
      </div>
      <i>Последнее обновление: 26 января 2025 10:00</i>
    </div>
  </div>
</main>

<style>
  .container {
    display: flex;
    gap: 16px;
    align-items: center;
    text-align: left;
  }
  .card {
    padding: 0;
    border-radius: 5px;
    /* border: 1px solid #c4c5cb; */
    background-color: rgba(255, 255, 255, .5);
    height: fit-content;
  }
  i {
    color: #888;
    font-size: 11px;
  }

</style>
