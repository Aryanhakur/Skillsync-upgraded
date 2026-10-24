from rest_framework import serializers


class JobSerializer(serializers.Serializer):
    id = serializers.CharField(required=False, allow_blank=True)
    title = serializers.CharField(required=False, allow_blank=True)
    company = serializers.CharField(required=False, allow_blank=True)
    location = serializers.CharField(required=False, allow_blank=True)
    description = serializers.CharField(required=False, allow_blank=True)
    Skills = serializers.CharField(required=False, allow_blank=True)
    Similarity_Score = serializers.FloatField(required=False, default=0.0)
    employment_type = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    job_apply_link = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    employer_logo = serializers.CharField(required=False, allow_null=True, allow_blank=True)


class CertificationSerializer(serializers.Serializer):
    name = serializers.CharField()
    url = serializers.URLField()
    skill = serializers.CharField(required=False, allow_blank=True)


class JobLinkSerializer(serializers.Serializer):
    jobProvider = serializers.CharField(required=False, allow_blank=True)
    url = serializers.CharField(required=False, allow_blank=True)


class CompanyTrendSkillSerializer(serializers.Serializer):
    skill = serializers.CharField()
    demand = serializers.IntegerField()


class CompanyTrendSerializer(serializers.Serializer):
    company = serializers.CharField()
    listings_analyzed = serializers.IntegerField()
    roles = serializers.ListField(child=serializers.CharField())
    top_skills = CompanyTrendSkillSerializer(many=True)
    missing_skills = serializers.ListField(child=serializers.CharField())


class MatchReportSerializer(serializers.Serializer):
    matched_skills = serializers.ListField(child=serializers.CharField())
    missing_skills = serializers.ListField(child=serializers.CharField())
    coverage_score = serializers.IntegerField()
    job_skills = serializers.ListField(child=serializers.CharField(), required=False)
