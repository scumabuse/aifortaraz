import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowRight,
  Camera,
  ClipboardCheck,
  Cpu,
  ShieldCheck,
} from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "../../api";
import { Button } from "../../components/ui/button";
import { estimateEfficiency, type EfficiencyInput } from "./efficiency";
import "./project.css";

export type ModelMetrics = {
  map_percent: number | null;
  fps: number | null;
  latency_ms: number | null;
};
export type ProjectInfo = {
  enterprise_name: string;
  model_metrics: ModelMetrics;
};
const inputs = [
  ["cameras", "Камер", 0, 10000, 1, 4],
  ["incidents", "Инцидентов с травмой в год", 0, 1000000, 1, 10],
  ["injuryCost", "Средняя стоимость одной травмы, ₸", 0, 1e12, 1, 500000],
  ["cameraCost", "Стоимость внедрения на камеру, ₸", 0, 1e12, 1, 150000],
  [
    "reductionPercent",
    "Предполагаемое снижение числа травм, %",
    0,
    100,
    0.1,
    20,
  ],
] as const;
const money = new Intl.NumberFormat("ru-KZ", {
  style: "currency",
  currency: "KZT",
  maximumFractionDigits: 0,
});
const number = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 });
const flow = [
  {
    name: "Камеры",
    info: "Передают видео на экран оператора и во внешний модуль ИИ.",
    icon: Camera,
  },
  {
    name: "Внешний ИИ",
    info: "Передаёт рамки, типы нарушений и снимки через API.",
    icon: Cpu,
  },
  {
    name: "Цифровой инспектор",
    info: "Показывает тревоги, сохраняет инциденты и рассчитывает показатели.",
    icon: ClipboardCheck,
  },
];
const comparisons = [
  [
    "Обнаружение",
    "Нарушение замечают во время обхода",
    "Событие поступает из ИИ по мере обнаружения",
  ],
  [
    "Реакция",
    "Оператор вручную сообщает мастеру",
    "Мастер получает уведомление и принимает инцидент",
  ],
  [
    "Отчётность",
    "Сведения собирают по разрозненным журналам",
    "История, снимки, время реакции и сводки доступны в одном месте",
  ],
];

export default function AboutPage() {
  const project = useQuery({
    queryKey: ["public-project"],
    queryFn: () => api<ProjectInfo>("/api/v1/public/project"),
    retry: false,
  });
  const [values, setValues] = useState<Record<keyof EfficiencyInput, string>>(
    () =>
      Object.fromEntries(
        inputs.map(([key, , , , , value]) => [key, String(value)]),
      ) as Record<keyof EfficiencyInput, string>,
  );
  const invalid = inputs.some(([key, , min, max, step]) => {
    const value = Number(values[key]);
    return (
      !values[key] ||
      value < min ||
      value > max ||
      ((key === "cameras" || key === "incidents") &&
        step === 1 &&
        !Number.isInteger(value))
    );
  });
  const estimate = invalid
    ? null
    : estimateEfficiency(
        Object.fromEntries(
          inputs.map(([key]) => [key, Number(values[key])]),
        ) as EfficiencyInput,
      );
  const metrics = project.data?.model_metrics;
  return (
    <main className="project-page">
      <header className="project-nav">
        <Link to="/" className="project-brand">
          <ShieldCheck size={25} />
          Цифровой инспектор ОТ
        </Link>
        <Button asChild>
          <Link to="/">
            К приложению <ArrowRight size={16} />
          </Link>
        </Button>
      </header>
      <section className="project-hero">
        <span className="eyebrow">Охрана труда на производстве</span>
        <h1>От нарушения на камере до реакции мастера</h1>
        <p>
          На участке оператору приходится следить сразу за несколькими камерами.
          Цифровой инспектор собирает события от внешнего ИИ в журнал, отмечает
          опасные эпизоды и фиксирует время реакции.
        </p>
      </section>
      <section className="project-section">
        <h2>Как работает система</h2>
        <ol className="project-flow">
          {flow.map(({ name, info, icon: Icon }) => (
            <li className="panel" key={name}>
              <Icon size={24} />
              <h3>{name}</h3>
              <p className="muted">{info}</p>
            </li>
          ))}
        </ol>
      </section>
      <section className="project-section">
        <h2>Метрики модели</h2>
        {project.isError && (
          <p role="alert">
            Метрики сейчас недоступны.{" "}
            <button className="accent" onClick={() => void project.refetch()}>
              Повторить
            </button>
          </p>
        )}
        <div className="project-flow">
          {(
            [
              ["mAP", metrics?.map_percent, "%"],
              ["Скорость", metrics?.fps, "FPS"],
              ["Задержка", metrics?.latency_ms, "мс"],
            ] as const
          ).map(([label, value, unit]) => (
            <div className="panel project-metric" key={label}>
              <span>{label}</span>
              <strong>
                {value == null ? "—" : `${number.format(value)} ${unit}`}
              </strong>
              {value == null && (
                <small className="muted">
                  {project.isPending
                    ? "Загрузка…"
                    : "Значение пока не опубликовано"}
                </small>
              )}
            </div>
          ))}
        </div>
      </section>
      <section className="panel project-section">
        <h2>Рабочий процесс</h2>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Шаг</th>
                <th>Ручное наблюдение</th>
                <th>С цифровым инспектором</th>
              </tr>
            </thead>
            <tbody>
              {comparisons.map(([step, before, after]) => (
                <tr key={step}>
                  <th scope="row">{step}</th>
                  <td>{before}</td>
                  <td>{after}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="panel project-section">
        <h2>Калькулятор эффективности</h2>
        <p className="muted">
          Проверьте сценарий для своего участка. Процент снижения травматизма
          задаётся как предположение.
        </p>
        <div className="project-calculator">
          <div className="edit-form">
            {inputs.map(([key, label, min, max, step]) => (
              <label key={key}>
                {label}
                <input
                  type="number"
                  min={min}
                  max={max}
                  step={step}
                  value={values[key]}
                  onChange={(e) =>
                    setValues((previous) => ({
                      ...previous,
                      [key]: e.target.value,
                    }))
                  }
                />
              </label>
            ))}
          </div>
          <div className="project-results" aria-live="polite">
            {!estimate ? (
              <p>
                Проверьте значения. Число камер и инцидентов должно быть целым.
              </p>
            ) : (
              <>
                <div>
                  <span>Ожидаемая экономия в год</span>
                  <strong>{money.format(estimate.annualSavings)}</strong>
                  <small>
                    Инциденты × доля предотвращённых × стоимость травмы
                  </small>
                </div>
                <div>
                  <span>Стоимость внедрения</span>
                  <strong>{money.format(estimate.investment)}</strong>
                  <small>Число камер × стоимость внедрения на камеру</small>
                </div>
                <div>
                  <span>Срок окупаемости</span>
                  <strong>
                    {estimate.paybackMonths == null
                      ? "—"
                      : `${number.format(estimate.paybackMonths)} мес.`}
                  </strong>
                  <small>
                    {estimate.paybackMonths == null
                      ? "При нулевой экономии срок не определяется"
                      : "12 × стоимость внедрения ÷ годовая экономия"}
                  </small>
                </div>
              </>
            )}
          </div>
        </div>
      </section>
      <section className="panel project-section">
        <h2>Команда проекта</h2>
        <ul className="project-team">
          <li>Интеграция и разработка приложения</li>
          <li>Компьютерное зрение и внешний модуль ИИ</li>
          <li>Охрана труда и проверка производственных сценариев</li>
        </ul>
      </section>
    </main>
  );
}
