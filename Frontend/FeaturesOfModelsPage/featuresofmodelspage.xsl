<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:template match="/">
        <html>
            <head>
                <title>Model Statistics</title>
                <style>
                    table { width: 100%; border-collapse: collapse; }
                    th, td { padding: 10px; text-align: left; border: 1px solid #ddd; }
                    th { background-color: #f2f2f2; }
                    h1, h2 { text-align: center; }
                </style>
            </head>
                <body>
                <h1>Stats Of Models Used In Sentiment Analysis</h1>
                <h3>Pages:</h3>
                <ul>
                    <li><button onclick="window.location.href='../MainPage/mainpage.html'">Main Page</button></li>
                    <li><button onclick="window.location.href='../DatasetAnalyzedPage/df_analyzed.xml'">DataFrame Analyzed</button></li>
                    <li><button onclick="window.location.href='../FeaturesOfModelsPage/Log_model_scores_df.xml'">Features Of Models</button></li>
                </ul>
                <!-- Logistic Regression Model -->
                <h2>Logistic Regression Model</h2>
                <Table border="1">
                    <tr bgcolor="#aaaaaa">
                        <th>Metric</th>
                        <th>precision</th>
                        <th>recall</th>
                        <th>f1-score</th>
                        <th>support</th>
                    </tr>
                    <xsl:for-each select="document('Log_model_scores_df.xml')/data/row">
                        <tr>
                            <td><xsl:value-of select="Metric"/></td>
                            <td><xsl:value-of select="precision"/></td>
                            <td><xsl:value-of select="recall"/></td>
                            <td><xsl:value-of select="f1-score"/></td>
                            <td><xsl:value-of select="support"/></td>
                        </tr>
                    </xsl:for-each>
                </Table>
                <!-- Random Forests Model -->     
                <h2>Random Forests Model</h2>
                <Table border="1">
                    <tr bgcolor="#aaaaaa">
                        <th>Metric</th>
                        <th>precision</th>
                        <th>recall</th>
                        <th>f1-score</th>
                        <th>support</th>
                    </tr>
                    <xsl:for-each select="document('rf_model_scores_df.xml')/data/row">
                        <tr>
                            <td><xsl:value-of select="Metric"/></td>
                            <td><xsl:value-of select="precision"/></td>
                            <td><xsl:value-of select="recall"/></td>
                            <td><xsl:value-of select="f1-score"/></td>
                            <td><xsl:value-of select="support"/></td>
                        </tr>
                    </xsl:for-each>
                </Table>
            </body>
        </html>
    </xsl:template>
</xsl:stylesheet>