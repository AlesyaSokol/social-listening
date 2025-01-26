<script>
  import TopicsList from './lib/TopicsList.svelte';
  import Map from './lib/Map.svelte';

  import data from './assets/topics_results.json';

  let topics = data;

  const selected = ["Сотрудничество Мурманской области с Белоруссией и развитие экологических инициатив.", 
  "Отключения электроэнергии и последствия сильного ветра в различных регионах России.",
  "Проблемы с замедлением и доступом к YouTube в России.",
  "Повышение пенсий и изменение условий кредитования для участников СВО.",
  "Распространение вируса Коксаки среди детей в детсадах и школах России вызывает тревогу родителей.",
  "Массовые драки и давка на мероприятиях в Москве и Ярославле, с пострадавшими и задержаниями.",
  "Масштабные рейды силовиков в московских клубах из-за подозрений в ЛГБТ-пропаганде и задержаниями.",
  "Российская вакцина против рака и меры по профилактике ВИЧ-инфекции в стране."
]

  topics = topics.filter(x => selected.includes(x.topic_gpt));
</script>

<main>
  <div class="container">
    <Map topics={topics} />
    <div class="card-container">
      <div class="card">
        <TopicsList topics={topics} />
      </div>
      <i>Последнее обновление: 26 января 2025 15:00</i>
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
    max-height: 500px;
    overflow-y: auto;
  }
  i {
    color: #888;
    font-size: 11px;
  }

</style>
