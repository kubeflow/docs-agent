{{/* Common labels stamped on every chart-managed resource. */}}
{{- define "gateway-guardrails.labels" -}}
app.kubernetes.io/part-of: docs-agent
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{/* Shared CORS policy — identical on every VirtualService route. */}}
{{- define "gateway-guardrails.corsPolicy" -}}
corsPolicy:
  allowOrigins:
{{ toYaml .Values.routing.cors.allowOrigins | indent 4 }}
  allowMethods:
{{ toYaml .Values.routing.cors.allowMethods | indent 4 }}
  allowHeaders:
{{ toYaml .Values.routing.cors.allowHeaders | indent 4 }}
  maxAge: {{ .Values.routing.cors.maxAge | quote }}
{{- end }}

{{/* FQDN of the session issuer service. */}}
{{- define "gateway-guardrails.issuerHost" -}}
session-issuer.{{ .Values.namespaces.docsAgent }}.svc.cluster.local
{{- end }}
