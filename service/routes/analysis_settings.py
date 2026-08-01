"""Public production analysis-settings capability endpoint."""

from service.analysis_settings import get_analysis_settings_capabilities
from service.models import AnalysisSettingsCapabilities


async def get_analysis_settings() -> AnalysisSettingsCapabilities:
    return get_analysis_settings_capabilities()
