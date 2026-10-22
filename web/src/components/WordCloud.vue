<script setup lang="ts">
/**
 * The most prominent terms of a selection, sized by how many resolutions each one finds. Words
 * are horizontal (rotated words are hard to read) and are links: following one lists those
 * resolutions. Placement is d3-cloud's spiral layout, recomputed when the terms or width change.
 */
import cloud from 'd3-cloud'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { RouterLink } from 'vue-router'
import type { CloudWord } from '@/charts/types'


const props = defineProps<{ words: CloudWord[]; height?: number }>()

type Item = cloud.Word & { word: CloudWord }

interface Placed {
  text: string
  size: number
  x: number
  y: number
  word: CloudWord
}

const container = ref<HTMLElement | null>(null)
const width = ref(800)
const placed = ref<Placed[]>([])
const height = () => props.height ?? 380
let layoutRun = 0

function layout() {
  const run = ++layoutRun
  const words = props.words
  if (!words.length) {
    placed.value = []
    return
  }
  const max = Math.max(...words.map((w) => w.count))
  const min = Math.min(...words.map((w) => w.count))
  // Area, not height, should follow the count: font size grows with its square root.
  const scale = (count: number) =>
    max === min ? 28 : 14 + (44 - 14) * Math.sqrt((count - min) / (max - min))
  const font = getComputedStyle(document.documentElement).getPropertyValue('--font-sans')

  cloud<Item>()
    .size([width.value, height()])
    .words(words.map((word): Item => ({ text: word.term, size: scale(word.count), word })))
    .padding(4)
    .rotate(0)
    .font(font)
    .fontWeight(500)
    .fontSize((d) => d.size!)
    .random(() => 0.5)
    .on('end', (out) => {
      if (run !== layoutRun) return
      placed.value = out.map((d) => ({
        text: d.text!,
        size: d.size!,
        x: d.x!,
        y: d.y!,
        word: d.word,
      }))
    })
    .start()
}

let observer: ResizeObserver | null = null
onMounted(() => {
  observer = new ResizeObserver((entries) => {
    const next = Math.round(entries[0]!.contentRect.width)
    if (next && Math.abs(next - width.value) > 8) {
      width.value = next
      layout()
    }
  })
  if (container.value) {
    width.value = container.value.clientWidth || 800
    observer.observe(container.value)
  }
  layout()
})
onBeforeUnmount(() => observer?.disconnect())
watch(() => props.words, layout)
</script>

<template>
  <div ref="container" class="cloud">
    <svg :width="width" :height="height()" :viewBox="`0 0 ${width} ${height()}`" role="list">
      <g :transform="`translate(${width / 2},${height() / 2})`">
        <RouterLink
          v-for="(p, i) in placed"
          :key="p.text"
          v-slot="{ href, navigate }"
          :to="p.word.to"
          custom
        >
          <a
            :href="href"
            role="listitem"
            class="word"
            :class="{ strong: i < 5 }"
            @click="navigate"
          >
            <title>{{ p.word.title }}</title>
            <text
              text-anchor="middle"
              :transform="`translate(${p.x},${p.y})`"
              :font-size="p.size"
            >
              {{ p.text }}
            </text>
          </a>
        </RouterLink>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.cloud {
  width: 100%;
  overflow: hidden;
}

svg {
  display: block;
}

.word text {
  fill: var(--ink-2);
  font-family: var(--font-sans);
  font-weight: 500;
}

.word.strong text {
  fill: var(--ink);
}

.word:hover text,
.word:focus-visible text {
  fill: var(--accent);
  text-decoration: underline;
}

.word:focus-visible {
  outline: none;
}
</style>
