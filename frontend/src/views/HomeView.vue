<template>
  <div class="home">
    <h1>PDF knowledge base</h1>
    <textarea id="query"></textarea>
    <div>
      <button @click="submit">Submit</button>
      <button @click="selectFiles">Select Files</button>
      <input @change="saveFiles" id="select-files" type="file" hidden multiple>
      <button @click="uploadPdfs">Upload PDFs</button>
    </div>
    <div>
      <div v-for="pdf in pdfs" :key="pdf.id">
        {{ pdf.name }}
      </div>
    </div>
  </div>
</template>

<script>
import { defineComponent } from 'vue';


export default defineComponent({
  name: 'HomeView',
  components: {},
  data() {
    return {
      query: '',
      pdfs: []
    }
  },
  methods: {
    submit() {
      console.log('submit')
    },
    saveFiles(e) {
      this.pdfs = e.target.files
    },
    uploadPdfs() {
      
    },
    selectFiles() {
      document.getElementById('select-files').click()
    }
  }
});
</script>
<style>
.home {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

textarea {
  resize: none;
  width: 300px;
  height: 150px;
}

button {
  margin: 5px;
}

</style>
